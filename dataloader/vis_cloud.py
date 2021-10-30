import copy
import numpy as np
import open3d as o3d
import read_data

if __name__ == "__main__":

    lidarfile_path = "/home/atas/poss_data/val/00/velodyne/000480.bin"
    gt_label_path = "/home/atas/poss_data/val/00/labels/000480.label"
    pred_label_path = "/home/atas/IROS21-FIDNet-SemanticKITTI/method_predictions/sequences/00/predictions/000480.label"

    points = read_data.read_points(lidarfile_path)
    labels = read_data.read_semlabels(gt_label_path)
    pred_labels = read_data.read_semlabels(pred_label_path)


    inv_label_dict = {0: 0,
                      10: 1,
                      11: 2,
                      15: 3,
                      18: 4,
                      20: 5,
                      30: 6,
                      31: 7,
                      32: 8,
                      40: 9,
                      44: 10,
                      48: 11,
                      49: 12,
                      50: 13,
                      51: 14,
                      70: 15,
                      71: 16,
                      72: 17,
                      80: 18,
                      81: 19}
    learning_map = {
        0: 0,
        1: 4,
        1: 5,
        2: 6,
        3: 7,
        4: 8,
        5: 9,
        6: 10,
        6: 11,
        6: 12,
        7: 13,
        8: 14,
        9: 15,
        10: 16,
        11: 17,
        12: 21,
        13: 22,
        18 : 0 # not sure where this comes from
        }

    gt_colors = []
    pred_color = []

    for i in range(0, len(pred_labels)):
        pred_labels[i] = inv_label_dict[pred_labels[i]]

    for i in pred_labels:
        pred_color.append(read_data.SEM_COLOR[learning_map[i]])
        
    for i in labels:
        gt_colors.append(read_data.SEM_COLOR[i])

    pcd= o3d.geometry.PointCloud()
    pcd.points= o3d.utility.Vector3dVector(np.asarray(points[:, 0:3]))
    pcd.colors= o3d.utility.Vector3dVector(np.asarray(gt_colors))
    #pcd.colors= o3d.utility.Vector3dVector(np.asarray(pred_color))

    o3d.visualization.draw_geometries([pcd])

    '''
    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    o3d.io.write_image("./sync.png", img)
    o3d.visualization.draw_geometries([img])
    '''
