import copy
import numpy as np
import open3d as o3d
import read_data
import sys
import numpy
import yaml
numpy.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":

    CFG = yaml.safe_load(
        open('/home/atas/mixed_data/semantickitti19.yaml', 'r'))

    color_dict = CFG["color_map"]

    lidarfile_path = "/home/atas/mixed_data/train/00/velodyne/000000.bin"
    #pred_label_path = "/home/atas/IROS21-FIDNet-SemanticKITTI/method_predictions/sequences/07/predictions/000460.label"
    pred_label_path = "/home/atas/mixed_data/train/00/labels/000000.label"

    points = read_data.read_points(lidarfile_path)
    pred_labels = read_data.read_semlabels(pred_label_path)

    pred_color = []

    for i in pred_labels:
        pred_color.append(color_dict[i])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points[:, 0:3]))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pred_color) / 255.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()
