
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from network.ResNet import *
import argparse
from torchvision.transforms import functional as FF
from dataloader.POSSDataset import *
from utils import *
from dataloader.laserscan import SemLaserScan, LaserScan
from dataloader.laserscan import read_points, calculate_normal, fill_spherical
from postproc.KNN import *
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs.msg._point_cloud2 as pc2
from ros2_numpy.ros2_numpy.point_cloud2 import *
import ros2_numpy

parser = argparse.ArgumentParser()

parser.add_argument('--range_y', dest="range_y", default=64, help="128")
parser.add_argument('--range_x', dest="range_x", default=512, help="2048")

# network settings
parser.add_argument('--backbone', dest="backbone", default="ResNet34_point",
                    help="ResNet34_aspp_1,ResNet34_aspp_2,ResNet_34_point")
parser.add_argument('--if_BN', dest="if_BN", default=True,
                    help="if use BN in the backbone net")
parser.add_argument('--if_remission', dest="if_remission",
                    default=True, help="if concatenate remmision in the input")
parser.add_argument('--if_range', dest="if_range", default=True,
                    help="if concatenate range in the input")

parser.add_argument('--with_normal', dest="with_normal",
                    default=True, help="if concatenate normal in the input")

parser.add_argument('--eval_epoch',  dest="eval_epoch", default=49,
                    help="0 or from the beginning, or from the middle")
parser.add_argument('--if_mixture',  dest="if_mixture",
                    default=True, help="if_mixture training")
parser.add_argument('--if_KNN',  dest="if_KNN", default=0,
                    help="0: no post; 1: original_knn; 2: our post")

fidnet_args = parser.parse_args()

inv_label_dict = {  # inverse of previous map
    0: 0,      # "unlabeled", and others ignored
    1: 10,     # "car"
    2: 11,     # "bicycle"
    3: 15,     # "motorcycle"
    4: 18,     # "truck"
    5: 20,     # "other-vehicle"
    6: 30,     # "person"
    7: 31,     # "bicyclist"
    8: 32,     # "motorcyclist"
    9: 40,     # "road"
    10: 44,    # "parking"
    11: 48,    # "sidewalk"
    12: 49,    # "other-ground"
    13: 50,    # "building"
    14: 51,    # "fence"
    15: 70,    # "vegetation"
    16: 71,    # "trunk"
    17: 72,    # "terrain"
    18: 80,    # "pole"
    19: 81     # "traffic-sign"
}


class FidnetRosNode(Node):
    def __init__(self, args):
        super().__init__("FidnetRosNode")

        root = '/home/atas/mixed_data/'
        CFG = yaml.safe_load(open(root+'semantickitti19.yaml', 'r'))
        self.color_dict = CFG["color_map"]

        self.args = args
        self.subscription = self.create_subscription(
            PointCloud2,
            '/ouster/points',
            self.callback,
            10)

        self.publisher = self.create_publisher(
            PointCloud2, '/ouster/points/segmented', 10)
        self.subscription  # prevent unused variable warning
        self.initialize = False

        if args.backbone == "ResNet34_aspp_1":
            Backend = resnet34_aspp_1(
                if_BN=args.if_BN, if_remission=args.if_remission, if_range=args.if_range)
            S_H = SemanticHead(20, 1152)
        if args.backbone == "ResNet34_aspp_2":
            Backend = resnet34_aspp_2(
                if_BN=args.if_BN, if_remission=args.if_remission, if_range=args.if_range)
            S_H = SemanticHead(20, 128*13)
        if args.backbone == "ResNet34_point":
            Backend = resnet34_point(if_BN=args.if_BN, if_remission=args.if_remission,
                                     if_range=args.if_range, with_normal=args.with_normal)
            S_H = SemanticHead(20, 1024)

        self.model = Final_Model(Backend, S_H)
        self.device = torch.device('cuda:{}'.format(0))
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(
            "/home/atas/IROS21-FIDNet-SemanticKITTI/save_semantic/ResNet34_point_512_64_BNTrue_remissionTrue_rangeTrue_normalTrue_rangemaskTrue_32_1.0_3.0_lr1_top_k0.15/299"))
        self.A = LaserScan(project=True, flip_sign=False, H=args.range_y,
                           W=args.range_x, fov_up=25.0, fov_down=-30.0)
        self.model.eval()

        scale_x = np.expand_dims(
            np.ones([args.range_y, args.range_x])*50.0, axis=-1).astype(np.float32)
        scale_y = np.expand_dims(
            np.ones([args.range_y, args.range_x])*50.0, axis=-1).astype(np.float32)
        scale_z = np.expand_dims(
            np.ones([args.range_y, args.range_x])*3.0, axis=-1).astype(np.float32)
        self.scale_matrx = np.concatenate([scale_x, scale_y, scale_z], axis=2)

    def callback(self, msg):
        a = time.time()

        pcd_as_numpy_array = np.array(
            list(read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)))

        points = pcd_as_numpy_array[:, 0:3]
        remissions = pcd_as_numpy_array[:, 3]
        self.A.set_points(points, remissions)

        xyz = torch.unsqueeze(FF.to_tensor(
            self.A.proj_xyz/self.scale_matrx), axis=0)
        remission = torch.unsqueeze(
            FF.to_tensor(self.A.proj_remission), axis=0)
        range_img = torch.unsqueeze(
            FF.to_tensor(self.A.proj_range/80.0), axis=0)

        if self.args.if_remission and not self.args.if_range:
            input_tensor = torch.cat([xyz, remission], axis=1)
        if self.args.if_remission and self.args.if_range:
            input_tensor = torch.cat([xyz, remission, range_img], axis=1)
        if not self.args.if_remission and not self.args.if_range:
            input_tensor = xyz

        if self.args.with_normal:
            normal_image = calculate_normal(
                fill_spherical(self.A.proj_range), self.args.range_y, self.args.range_x, 25.0, -30.0)
            normal_image = normal_image * \
                np.transpose(
                    [self.A.proj_mask, self.A.proj_mask, self.A.proj_mask], [1, 2, 0])
            normal_image = torch.unsqueeze(FF.to_tensor(
                normal_image.astype(np.float32)), axis=0)
            input_tensor = torch.cat([input_tensor, normal_image], axis=1)
        input_tensor = input_tensor.to(self.device)
        with torch.cuda.amp.autocast(enabled=self.args.if_mixture):
            semantic_output = self.model(input_tensor)

        semantic_pred = get_semantic_segmentation(semantic_output[:1, :, :, :])

        if self.args.if_KNN == 2:
            t_1 = torch.squeeze(range_img*80.0).detach().to(self.device)
            t_3 = torch.squeeze(semantic_pred).detach().to(self.device)

            proj_unfold_range, proj_unfold_pre = NN_filter(t_1, t_3)

            semantic_pred = np.squeeze(semantic_pred.detach().cpu().numpy())
            proj_unfold_range = proj_unfold_range.cpu().numpy()
            proj_unfold_pre = proj_unfold_pre.cpu().numpy()
            label = []
            for jj in range(len(self.A.proj_x)):
                y_range, x_range = self.A.proj_y[jj], self.A.proj_x[jj]
                upper_half = 0
                if y_range < 0:
                    y_range = 0
                if x_range < 0:
                    x_range = 0
                if self.A.unproj_range[jj] == self.A.proj_range[y_range, x_range]:
                    lower_half = inv_label_dict[semantic_pred[y_range, x_range]]
                else:
                    potential_label = proj_unfold_pre[0, :, y_range, x_range]
                    potential_range = proj_unfold_range[0, :, y_range, x_range]
                    min_arg = np.argmin(
                        abs(potential_range-self.A.unproj_range[jj]))
                    lower_half = inv_label_dict[potential_label[min_arg]]
                label_each = (upper_half << 16) + lower_half
                label.append(label_each)

        if self.args.if_KNN == 0:
            semantic_pred = np.squeeze(semantic_pred.detach().cpu().numpy())
            label = []
            for jj in range(len(self.A.proj_x)):
                y_range, x_range = self.A.proj_y[jj], self.A.proj_x[jj]
                upper_half = 0
                if y_range < 0:
                    y_range = 0
                if x_range < 0:
                    x_range = 0
                lower_half = inv_label_dict[semantic_pred[y_range, x_range]]
                label_each = (upper_half << 16) + lower_half
                label.append(label_each)

        label = np.asarray(label)
        label = label.astype(np.uint32)
        # label.tofile(label_file)
        b = time.time()

        print("processed a frame with fidnet took: ", b-a)

        points_arr = np.zeros((len(points),), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('r', np.uint8),
            ('g', np.uint8),
            ('b', np.uint8),
            ('label', np.uint8)])

        points_arr['x'] = points[:, 0]
        points_arr['y'] = points[:, 1]
        points_arr['z'] = points[:, 2]

        r, g, b = [], [], []
        label_array = []

        for l in label:
            r.append(self.color_dict[l][0])
            g.append(self.color_dict[l][1])
            b.append(self.color_dict[l][2])
            label_array.append(l)

        points_arr['r'] = r
        points_arr['g'] = g
        points_arr['b'] = b
        points_arr['label'] = label_array

        points_arr = merge_rgb_fields(points_arr)
        new_msg = array_to_pointcloud2(
            points_arr, msg.header.stamp, msg.header.frame_id)

        new_msg = ros2_numpy.ros2_numpy.msgify(PointCloud2, points_arr)
        new_msg.header = msg.header

        self.publisher.publish(new_msg)


def main(args=None):

    rclpy.init(args=args)
    node = FidnetRosNode(fidnet_args)

    while rclpy.ok():
        rclpy.spin_once(node)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(args=None)
