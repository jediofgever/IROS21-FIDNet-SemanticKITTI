
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
from postproc.KNN import KNN


parser = argparse.ArgumentParser()
# parameters for dataset
parser.add_argument('--dataset', dest="dataset",
                    default='POSSDataset', help='')
parser.add_argument('--root',  dest="root",
                    default='/home/atas/mixed_data/', help="poss_data/")
parser.add_argument('--range_y', dest="range_y", default=64, help="64")
parser.add_argument('--range_x', dest="range_x", default=512, help="512")
#parser.add_argument('--code_mode', dest= "code_mode", default="train", help="train or val")


# network settings
parser.add_argument('--backbone', dest="backbone", default="ResNet34_point",
                    help="ResNet34_aspp_1,ResNet34_aspp_2,ResNet_34_point")
parser.add_argument('--batch_size', dest="batch_size", default=1, help="bs")
parser.add_argument('--if_BN', dest="if_BN", default=True,
                    help="if use BN in the backbone net")
parser.add_argument('--if_remission', dest="if_remission",
                    default=True, help="if concatenate remmision in the input")
parser.add_argument('--if_range', dest="if_range", default=True,
                    help="if concatenate range in the input")
parser.add_argument('--if_range_mask', dest="if_range_mask",
                    default=True, help="if if_range_mask")
parser.add_argument('--with_normal', dest="with_normal",
                    default=True, help="if concatenate normal in the input")


# training settins
parser.add_argument('--eval_epoch',  dest="eval_epoch", default=299,
                    help="0 or from the beginning, or from the middle")
parser.add_argument('--lr_policy',  dest="lr_policy",
                    default=1, help="lr_policy: 1, 2 or 0")
parser.add_argument('--weight_WCE',  dest="weight_WCE",
                    default=1.0, help="weight_WCE")
parser.add_argument('--weight_LS',  dest="weight_LS",
                    default=3.0, help="weight_LS")
parser.add_argument('--top_k_percent_pixels',  dest="top_k_percent_pixels",
                    default=0.15, help="top_k_percent_pixels, hard mining")
parser.add_argument('--BN_train',  dest="BN_train", default=True,
                    help="if BN_train, false when batch_size is small")
parser.add_argument('--if_mixture',  dest="if_mixture",
                    default=True, help="if_mixture training")
parser.add_argument('--if_KNN',  dest="if_KNN", default=2,
                    help="0: no post; 1: original_knn; 2: our post")

args = parser.parse_args()

learning_map_inv = {  # inverse of previous map
    0: 0,     # "unlabeled", and others ignored
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
    11: 48,   # "sidewalk"
    12: 49,    # "other-ground"
    13: 50,    # "building"
    14: 51,    # "fence"
    15: 70,    # "vegetation"
    16: 71,    # "trunk"
    17: 72,    # "terrain"
    18: 80,    # "pole"
    19: 81    # "traffic-sign"
}

dataset_train = POSSDataset(root=args.root, split='test', is_train=False, range_img_size=(args.range_y, args.range_x), if_aug='True',
                            if_range_mask=args.if_range_mask, if_remission=args.if_remission, if_range=args.if_range, with_normal=args.with_normal)

save_path = "/home/atas/IROS21-FIDNet-SemanticKITTI/save_semantic/"
temp_path = args.backbone+"_"+str(args.range_x)+"_"+str(args.range_y)+"_BN"+str(args.if_BN)+"_remission"+str(args.if_remission)+"_range"+str(args.if_range)+"_normal"+str(
    args.with_normal)+"_rangemask"+str(args.if_range_mask)+"_"+str(32)+"_"+str(args.weight_WCE)+"_"+str(args.weight_LS)+"_lr"+str(args.lr_policy)+"_top_k"+str(args.top_k_percent_pixels)
save_path = save_path+temp_path+"/"
#save_path = "/home/atas/IROS21-FIDNet-SemanticKITTI/save_semantic/kitti/"

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

model = Final_Model(Backend, S_H)
device = torch.device('cuda:{}'.format(0))

model.to(device)

model.load_state_dict(torch.load(save_path+str(args.eval_epoch)))

print("Going to evaluate with these weights: ", save_path+str(args.eval_epoch))

scale_x = np.expand_dims(
    np.ones([args.range_y, args.range_x])*50.0, axis=-1).astype(np.float32)
scale_y = np.expand_dims(
    np.ones([args.range_y, args.range_x])*50.0, axis=-1).astype(np.float32)
scale_z = np.expand_dims(
    np.ones([args.range_y, args.range_x])*3.0, axis=-1).astype(np.float32)
scale_matrx = np.concatenate([scale_x, scale_y, scale_z], axis=2)


A = LaserScan(project=True, flip_sign=False, H=args.range_y,
              W=args.range_x, fov_up=20.0, fov_down=-25.0)

model.eval()


if args.if_KNN == 1:
    knn_params = {'knn': 5, 'search': 5, 'sigma': 1.0, 'cutoff': 1.0}
    post_knn = KNN(knn_params, 20)

all_seq_list = ['07']

if not os.path.exists("./method_predictions/"):
    os.mkdir("./method_predictions/")
if not os.path.exists("./method_predictions/sequences/"):
    os.mkdir("./method_predictions/sequences/")

for seq_name in all_seq_list:

    lidar_list = glob.glob(args.root+'test/'+seq_name+'/*/*.bin')

    save_path_for_prediction = "./method_predictions/sequences/"+seq_name+"/"
    if not os.path.exists(save_path_for_prediction):
        os.mkdir(save_path_for_prediction)

    save_path_for_prediction = "./method_predictions/sequences/"+seq_name+"/predictions/"
    if not os.path.exists(save_path_for_prediction):
        os.mkdir(save_path_for_prediction)

    for i in range(len(lidar_list)):
        if i % 100 == 0:
            print(i)
        path_list = lidar_list[i].split('/')
        label_file = save_path_for_prediction + \
            path_list[-1][:len(path_list[-1])-3]+"label"
        if os.path.exists(label_file):
            os.remove(label_file)
        A.open_scan(lidar_list[i])

        #xyz = torch.unsqueeze(FF.to_tensor(np.expand_dims(A.proj_mask,axis=-1)*(A.proj_xyz-xyz_mean)/xyz_std),axis=0)
        xyz = torch.unsqueeze(FF.to_tensor(A.proj_xyz/scale_matrx), axis=0)

        #remission = torch.unsqueeze(FF.to_tensor(A.proj_mask*(A.proj_remission-remission_mean)/remission_std),axis=0)
        remission = torch.unsqueeze(FF.to_tensor(A.proj_remission), axis=0)

        range_img = torch.unsqueeze(FF.to_tensor(A.proj_range/80.0), axis=0)

        if args.if_remission and not args.if_range:
            input_tensor = torch.cat([xyz, remission], axis=1)
        if args.if_remission and args.if_range:
            input_tensor = torch.cat([xyz, remission, range_img], axis=1)
        if not args.if_remission and not args.if_range:
            input_tensor = xyz
        if args.with_normal:
            normal_image = dataset_train.calculate_normal(
                dataset_train.fill_spherical(A.proj_range))
            normal_image = normal_image * \
                np.transpose(
                    [A.proj_mask, A.proj_mask, A.proj_mask], [1, 2, 0])
            normal_image = torch.unsqueeze(FF.to_tensor(
                normal_image.astype(np.float32)), axis=0)
            input_tensor = torch.cat([input_tensor, normal_image], axis=1)
        input_tensor = input_tensor.to(device)
        with torch.cuda.amp.autocast(enabled=args.if_mixture):
            semantic_output = model(input_tensor)
        semantic_pred = get_semantic_segmentation(semantic_output[:1, :, :, :])

        if args.if_KNN == 2:
            t_1 = torch.squeeze(range_img*80.0).detach().to(device)
            t_3 = torch.squeeze(semantic_pred).detach().to(device)

            proj_unfold_range, proj_unfold_pre = NN_filter(t_1, t_3)

            semantic_pred = np.squeeze(semantic_pred.detach().cpu().numpy())
            proj_unfold_range = proj_unfold_range.cpu().numpy()
            proj_unfold_pre = proj_unfold_pre.cpu().numpy()
            label = []
            for jj in range(len(A.proj_x)):
                y_range, x_range = A.proj_y[jj], A.proj_x[jj]
                upper_half = 0
                if y_range < 0:
                    y_range = 0
                if x_range < 0:
                    x_range = 0
                if A.unproj_range[jj] == A.proj_range[y_range, x_range]:
                    lower_half = learning_map_inv[semantic_pred[y_range, x_range]]
                else:
                    potential_label = proj_unfold_pre[0, :, y_range, x_range]
                    potential_range = proj_unfold_range[0, :, y_range, x_range]
                    min_arg = np.argmin(
                        abs(potential_range-A.unproj_range[jj]))
                    lower_half = learning_map_inv[potential_label[min_arg]]
                label_each = (upper_half << 16) + lower_half
                label.append(label_each)

        if args.if_KNN == 1:

            unproj_argmax = post_knn(torch.squeeze(range_img*80.0).detach().cpu(), torch.squeeze(FF.to_tensor(np.reshape(A.unproj_range, (1, -1)))).detach().cpu(), torch.squeeze(semantic_pred).detach().cpu(
            ), torch.squeeze(FF.to_tensor(np.reshape(A.proj_x, (1, -1)))).to(dtype=torch.long).detach().cpu(), torch.squeeze(FF.to_tensor(np.reshape(A.proj_y, (1, -1)))).to(dtype=torch.long).detach().cpu())

            label = []
            for i in unproj_argmax:
                upper_half = 0
                lower_half = learning_map_inv[i.item()]
                label_each = (upper_half << 16) + lower_half
                label.append(label_each)

        if args.if_KNN == 0:
            semantic_pred = np.squeeze(semantic_pred.detach().cpu().numpy())
            label = []
            for jj in range(len(A.proj_x)):
                y_range, x_range = A.proj_y[jj], A.proj_x[jj]
                upper_half = 0
                if y_range < 0:
                    y_range = 0
                if x_range < 0:
                    x_range = 0
                lower_half = learning_map_inv[semantic_pred[y_range, x_range]]
                label_each = (upper_half << 16) + lower_half
                label.append(label_each)

        label = np.asarray(label)
        label = label.astype(np.uint32)
        label.tofile(label_file)
