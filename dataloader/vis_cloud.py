import copy
import numpy as np
import open3d as o3d
import read_data
import sys
import numpy
import yaml
import glob

if __name__ == "__main__":

    root = '/home/atas/mixed_data/'

    CFG = yaml.safe_load(open(root+'semantickitti19.yaml', 'r'))
    color_dict = CFG["color_map"]

    lidar_list = sorted(glob.glob(root + "test/*/*/*.bin"))
    gt_label_list = [i.replace("velodyne", "labels") for i in lidar_list]
    gt_label_list = [i.replace("bin", "label") for i in gt_label_list]

    pred_label_list = sorted(glob.glob(
        "/home/atas/IROS21-FIDNet-SemanticKITTI/method_predictions/sequences/*/*/*.label"))

    vis_gt = o3d.visualization.Visualizer()
    vis_pred = o3d.visualization.Visualizer()

    vis_gt.create_window(window_name='TopLeft', width=1000, height=1200, left=0, top=0)
    vis_pred.create_window(window_name='TopRight', width=1000, height=1200, left=1000, top=0)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()

    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])

    for index, val in enumerate(lidar_list):

        points = read_data.read_points(lidar_list[index])
        pred_labels = read_data.read_semlabels(pred_label_list[index])
        gt_labels = read_data.read_semlabels(gt_label_list[index])
        pred_color = []
        gt_color = []

        for i, k in enumerate(pred_labels):
            pred_color.append(color_dict[pred_labels[i]])
            gt_color.append(color_dict[gt_labels[i]])

        pcd_gt.points = o3d.utility.Vector3dVector(np.asarray(points[:, 0:3]))
        pcd_pred.points = o3d.utility.Vector3dVector(np.asarray(points[:, 0:3]))

        pcd_gt.colors = o3d.utility.Vector3dVector(np.asarray(gt_color) / 255.0)
        pcd_pred.colors = o3d.utility.Vector3dVector(np.asarray(pred_color) / 255.0)

        if index == 0:
            vis_gt.add_geometry(pcd_gt)
            vis_pred.add_geometry(pcd_pred)

        vis_gt.update_geometry(pcd_gt)
        vis_gt.poll_events()
        vis_gt.update_renderer()

        vis_pred.update_geometry(pcd_pred)
        vis_pred.poll_events()
        vis_pred.update_renderer()

    vis_gt.destroy_window()
    vis_pred.destroy_window()
