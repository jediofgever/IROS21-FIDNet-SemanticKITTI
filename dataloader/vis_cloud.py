import copy
import numpy as np
import open3d as o3d
import read_data
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":

    lidarfile_path = "/home/atas/poss_data/test/07/velodyne/000500.bin"
    
    #gt_label_path = "/home/atas/poss_data/test/07/labels/000000.label"
    pred_label_path = "/home/atas/IROS21-FIDNet-SemanticKITTI/method_predictions/sequences/07/predictions/000500.label"

    points = read_data.read_points(lidarfile_path)
    
    #labels = read_data.read_semlabels(gt_label_path)
    pred_labels = read_data.read_semlabels(pred_label_path)

    gt_colors = []
    pred_color = []

    for i in pred_labels:
        pred_color.append(read_data.SEM_COLOR[i])
        
    #for i in labels:
    #    gt_colors.append(read_data.SEM_COLOR[i])
    
    print(len(np.asarray(points[:, 0:3])))
    print(len(pred_labels))

    pcd= o3d.geometry.PointCloud()
    pcd.points= o3d.utility.Vector3dVector(np.asarray(points[:, 0:3]))
    #pcd.colors= o3d.utility.Vector3dVector(np.asarray(gt_colors))
    pcd.colors= o3d.utility.Vector3dVector(np.asarray(pred_color))
    
    o3d.visualization.draw_geometries([pcd])

    '''
    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    o3d.io.write_image("./sync.png", img)
    o3d.visualization.draw_geometries([img])
    '''
