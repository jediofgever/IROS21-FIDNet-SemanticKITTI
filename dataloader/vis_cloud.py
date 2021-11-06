import copy
import numpy as np
import open3d as o3d
import read_data
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":

    lidarfile_path = "/home/atas/IROS21-FIDNet-SemanticKITTI/poss_data/test/07/velodyne/000001.bin"
    pred_label_path = "/home/atas/IROS21-FIDNet-SemanticKITTI/method_predictions/sequences/07/predictions/000001.label"

    #lidarfile_path_poss = "/home/atas/IROS21-FIDNet-SemanticKITTI/poss_data/test/04/velodyne/000002.bin"
    #gt_label_path = "/home/atas/IROS21-FIDNet-SemanticKITTI/method_predictions/sequences/04/predictions/000002.label"
    
    points = read_data.read_points(lidarfile_path)
    #points_poss = read_data.read_points(lidarfile_path_poss)

    pred_labels = read_data.read_semlabels(pred_label_path)
    #labels = read_data.read_semlabels(gt_label_path)

    pred_color = []
    gt_colors = []

    for i in pred_labels:
        pred_color.append(read_data.SEM_COLOR[i])

    #for i in labels:
    #    gt_colors.append(read_data.SEM_COLOR[i])
    
    print(len(pred_labels))
    print(len(points))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points[:, 0:3]))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pred_color))

    pcd_poss = o3d.geometry.PointCloud()
    #pcd_poss.points = o3d.utility.Vector3dVector(np.asarray(points_poss[:, 0:3]))
    #pcd_poss.colors = o3d.utility.Vector3dVector(np.asarray(gt_colors))

    o3d.visualization.draw_geometries([  pcd])
