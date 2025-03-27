from tools import pretrain_run_net as pretrain
from tools import test_svm_run_net_modelnet40 as test_svm_modelnet40
from tools import test_svm_run_net_scan as test_svm_scan
from tools import finetune_run_net as finetune
from tools import test_run_net as test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter
from models.Point_M2AE import Point_M2AE
import yaml
import argparse
from types import SimpleNamespace
import numpy as np
import re
import open3d as o3d
import datasets
from datasets.bagle import bagle
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd



def save_point_cloud(points, filename):
    """将点云数据保存为PLY或TXT文件"""
    # 确保points是numpy数组
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    # 确保points的形状为(N, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云数据应该是一个形状为(N, 3)的数组")

    # 转换数据类型为float64，如果需要的话
    if points.dtype != np.float64:
        points = points.astype(np.float64)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if filename.endswith('.ply'):
        o3d.io.write_point_cloud(filename, pc)
    else:
        np.savetxt(filename, points, fmt='%f')

        
def find_closest_points_in_original(original_points,reconstructed_points ):
    # 计算每个重建点与原始点云中所有点之间的距离
    dists = torch.cdist( original_points,reconstructed_points, p=2)
    _, min_indices = torch.min(dists, dim=1)
    return min_indices
   
def find_largest_difference_among_closest(reconstructed_points, original_points, closest_indices,k):
    # 从原始点云中选择与重建点云最接近的点
    closest_points = reconstructed_points[closest_indices]

    # 计算这些最近点与其对应的重建点之间的距离
    dists = torch.norm( closest_points-original_points , dim=1)

    # 找到这些距离中的最大值
    max_dist, max_index = torch.topk(dists, k)
    return closest_indices[max_index], max_dist
       
        
def create_colored_ply_file(original_points, reconstructed_points,k, output_file):
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    closest_indices = find_closest_points_in_original(reconstructed_points, original_points)
    largest_diff_indices, _ = find_largest_difference_among_closest(reconstructed_points, original_points, closest_indices, k)

    original_points_np = original_points.numpy()
    colors = np.ones((len(original_points_np), 3))  # 默认白色（1, 1, 1）

    # 将差异最大的k个点设置为红色
    for idx in largest_diff_indices:
        colors[idx] = [1, 0, 0]  # 红色

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Colored point cloud with top {k} differences saved to '{output_file}'")
    

def create_heatmap_ply(original_points, rec_points, output_file):
    
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    
    # 计算距离
    dists = torch.cdist(original_points, rec_points, p=2)
    min_dists, _ = torch.min(dists, dim=1)
    
    # 标准化距离
    min_dists_np = min_dists.numpy()
    min_dists_normalized = (min_dists_np - np.min(min_dists_np)) / (np.max(min_dists_np) - np.min(min_dists_np))

    print("距离标准化：最小值 =", np.min(min_dists_normalized), "最大值 =", np.max(min_dists_normalized))
    
    # 生成颜色
    colors = plt.cm.gray(min_dists_normalized)[:, :3]  # 取RGB，忽略透明度

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_points.numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存点云到PLY文件
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Heatmap PLY file saved to '{output_file}'") 
    
    # 创建一个 Namespace 对象，模拟命令行参数
args = SimpleNamespace()
args.config = 'cfgs/pre-training/point-m2ae.yaml'  # 配置文件路径
args.resume = False
args.local_rank = 0
args.experiment_path = 'experiments'  # 根据需要设置实验路径
# 加载配置
config = get_config(args)

# 确保实验路径存在
os.makedirs(args.experiment_path, exist_ok=True)


model = Point_M2AE(config.model)

state_dict = torch.load('p_m2ae_carrot_8192_newbest.pth')


new_state_dict = OrderedDict()
for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
    name = k[7:] # 截取`module.`后面的xxx.weight
    new_state_dict[name] = v


model.load_state_dict(new_state_dict)

model = model.to('cuda')
model.eval()


dataset = bagle(data_path="../mvtec3d/potato/vis")
if len(dataset) == 0:
    print(f"警告: 数据集 {data_path} 为空，跳过处理。")
    
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


#taxonomy_id, model_id, points = next(iter(data_loader))

for i, (taxonomy_id, model_id, points) in enumerate(data_loader):



    # 转移到 GPU
    points = points.to('cuda')
    
    loss,original_points,gt_points,rec_points,target_layer_points,center_points,mas_points,absolute_rec_points,dismas_points,all_features = model(points)
    
    
    for i, features in enumerate(all_features):
        print(f'Layer {i} features shape: {features.shape}')
    '''
    center_points = center_points.detach().cpu()
    center_points = center_points.reshape(-1, 3)
    save_point_cloud(center_points, 'center.ply')

   
    
    absolute_rec_points = absolute_rec_points.detach().cpu()
    absolute_rec_points = absolute_rec_points.reshape(-1, 3)
    save_point_cloud(absolute_rec_points, 'rec.ply')
    
    mas_points = mas_points.detach().cpu()
    mas_points= mas_points.reshape(-1, 3)
    save_point_cloud(mas_points, 'mas.ply')
    '''

    
    
    
    
    
    
    
    
    
    
    
    
    '''
    original_points1 = original_points1.detach().cpu()
    mas_points1 = mas_points1.detach().cpu()
    rec_points1 = absolute_rec_points1.detach().cpu()
    dismas_points1=dismas_points1.detach().cpu()
    original_points2 = original_points2.detach().cpu()
    mas_points2 = mas_points2.detach().cpu()
    rec_points2 = absolute_rec_points2.detach().cpu()
    dismas_points2=dismas_points2.detach().cpu()
    
    
    
    
    
    
    
    
    original_points1 = original_points1.reshape(-1, 3)
    mas_points1 = mas_points1.reshape(-1, 3)
    rec_points1 = rec_points1.reshape(-1, 3)
    dismas_points1=dismas_points1.reshape(-1, 3)
    original_points2 = original_points2.reshape(-1, 3)
    mas_points2 = mas_points2.reshape(-1, 3)
    rec_points2 = rec_points2.reshape(-1, 3)
    dismas_points2=dismas_points2.reshape(-1, 3)
    
    
    allrec = np.concatenate((rec_points1,rec_points2),axis=0)
    allrec = torch.tensor(allrec)
    
    
    
    
    save_point_cloud(original_points1, 'ori1.ply')
    save_point_cloud(mas_points1, 'mas1.ply')
    save_point_cloud(rec_points1, 'abrecs1.ply')
    save_point_cloud(dismas_points1, 'dismass1.ply')
    save_point_cloud(original_points2, 'ori2.ply')
    save_point_cloud(mas_points2, 'mas2.ply')
    save_point_cloud(rec_points2, 'abrecs2.ply')
    save_point_cloud(dismas_points2, 'dismass2.ply')
    save_point_cloud(allrec, 'allrec.ply')
    
    '''
        
        
       
       












