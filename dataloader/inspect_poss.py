import copy
import numpy as np
import open3d as o3d
import read_data
import sys
import numpy
import yaml
import glob
from laserscan import SemLaserScan, LaserScan

poss2kitti = {
    0: 0,
    4: 30,
    5: 30,
    7: 10,
    9: 70,
    10: 70,
    15: 50,
    22: 40
}

def LabelPOSS2KITTI(root="/home/atas/poss_dataset/", split="train"):

    lidar_list = sorted(glob.glob(root+split+'/*/*/*.bin'))
    label_list = [i.replace("velodyne", "labels") for i in lidar_list]
    label_list = [i.replace("bin", "label") for i in label_list]

    CFG = yaml.safe_load(
        open('/home/atas/IROS21-FIDNet-SemanticKITTI/dataloader/poss.yaml', 'r'))

    color_dict = CFG["color_map"]
    nclasses = 14
    A = SemLaserScan(nclasses=nclasses, sem_color_dict=color_dict)

    print("handling ", len(label_list), "labels")

    for index_file, l in enumerate(label_list):

        A.open_scan(lidar_list[index_file])
        A.open_label(label_list[index_file])

        current_ins_labels = A.inst_label
        current_sem_labels = A.sem_label
        
        if index_file %50 == 0 :
            print ("Completed ", index_file)
        
        new_labels = []
        # translate from poss to kitti semantic labels
        for index, p in enumerate(current_sem_labels):
            
            ins_label = current_ins_labels[index]
            translated_sem_label = poss2kitti.get(p)
            
            if translated_sem_label:
                current_sem_labels[index] = translated_sem_label
            else:
                translated_sem_label = 0
                current_sem_labels[index] = translated_sem_label
            
            label_each = (ins_label << 16) + translated_sem_label
            new_labels.append(label_each)
            
        new_labels = np.asarray(new_labels)
        new_labels = new_labels.astype(np.uint32)
        new_labels.tofile(label_list[index_file])    


if __name__ == "__main__":

    CFG = yaml.safe_load(
        open('/home/atas/IROS21-FIDNet-SemanticKITTI/dataloader/poss.yaml', 'r'))

    color_dict = CFG["color_map"]

    lidarfile_path = "/home/atas/poss_dataset/train/00/velodyne/000460.bin"
    pred_label_path = "/home/atas/poss_dataset/train/00/labels/000460.label"

    points = read_data.read_points(lidarfile_path)
    pred_labels = read_data.read_semlabels(pred_label_path)

    pred_color = []

    #LabelPOSS2KITTI()

    print(np.unique(pred_labels))

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
