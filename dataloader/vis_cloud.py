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

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()

    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])

    for index, val in enumerate(lidar_list):

        print("Visualizing ", lidar_list[index])
        print("Visualizing ", pred_label_list[index])

        points = read_data.read_points(lidar_list[index])
        pred_labels = read_data.read_semlabels(pred_label_list[index])
        gt_labels = read_data.read_semlabels(gt_label_list[index])
        pred_color = []
        gt_color = []

        for i, k in enumerate(pred_labels):
            pred_color.append(color_dict[pred_labels[i]])
            gt_color.append(color_dict[gt_labels[i]])

        pcd.points = o3d.utility.Vector3dVector(np.asarray(points[:, 0:3]))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(gt_color) / 255.0)

        if index == 0:
            vis.add_geometry(pcd)

        vis.update_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
