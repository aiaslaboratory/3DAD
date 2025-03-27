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
from models.Point_M2AE import Point_M2AE,H_Encoder
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
import torch.nn as nn
from collections import OrderedDict
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from matplotlib import cm
import plyfile



def save_ply(filename, points, anomaly_scores):
    # 将异常分数转换为颜色（这里简单映射到红色通道）
    # 您可能需要根据异常分数的范围调整这里的映射
    
    
    points = points.reshape(-1, 3)
    anomaly_scores = anomaly_scores.flatten()
    # 初始化颜色数组
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    
    threshold = np.percentile(anomaly_scores, 80)
    #threshold = 0.8
    condition = anomaly_scores > threshold
    
    # 设置异常分数高于 0.8 的点为红色
    colors[condition, 0] = 255  # 红色通道
    # 对于其他点，设置为灰色（您可以根据需要调整这个颜色）
    colors[~condition] = [128, 128, 128]  # 灰色
  
    vertices = np.array([tuple(point) + tuple(color) for point, color in zip(points, colors)], 
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    ply_data = PlyData([PlyElement.describe(vertices, 'vertex')], text=True)
    ply_data.write(filename)
    '''
    
    
    # 初始化颜色数组
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    
    # 定义一个从冷到热的颜色映射，这里简化为蓝色到红色的过渡
    # 异常分数较低时偏蓝（冷色调），较高时偏红（暖色调）
    colors[:, 0] = np.clip(anomaly_scores * 255, 0, 255)  # 红色通道，异常分数高时增强
    colors[:, 2] = np.clip((1 - anomaly_scores) * 255, 0, 255)  # 蓝色通道，异常分数低时增强
    
    # 构造顶点数据结构
    vertices = np.array([tuple(point) + tuple(color) for point, color in zip(points, colors)],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    # 创建并写入PLY文件
    ply_data = plyfile.PlyData([plyfile.PlyElement.describe(vertices, 'vertex')], text=True)
    ply_data.write(filename)
    '''
def compute_neighbor_distances(pcd, k=10):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distances = []

    for point in pcd.points:
        [_, idx, dist] = pcd_tree.search_knn_vector_3d(point, k)
        distances.append(np.mean(dist))

    return np.array(distances)


args = SimpleNamespace()
args.teacher_config = 'cfgs/pre-training/point-m2ae.yaml'  # 教师模型配置文件路径
args.student_config = 'cfgs/pre-training/point-m2ae-4.yaml'  # 学生模型配置文件路径
args.resume = False  # 保留你原有的其他参数设置
args.local_rank = 0
args.experiment_path = 'experiments'  # 根据需要设置实验路径

# 加载教师模型配置
teacher_args = SimpleNamespace(**vars(args))  # 创建一个新的args副本
teacher_args.config = args.teacher_config  # 更新配置文件路径
teacher_config = get_config(teacher_args)  # 使用更新后的args加载配置

# 加载学生模型配置
student_args = SimpleNamespace(**vars(args))  # 再次创建args的副本以保持原始args不变
student_args.config = args.student_config  # 更新配置文件路径
student_config = get_config(student_args)  # 使用更新后的args加载配置

# 确保实验路径存在
os.makedirs(args.experiment_path, exist_ok=True)
# 初始化模型
teacher_model = H_Encoder(teacher_config.model)
student_model = H_Encoder(student_config.model)
teacher_model = teacher_model.to('cuda')
student_model = student_model.to('cuda')
teacher_model.eval()
student_model.eval()
state_dict = torch.load('m2ae_bagel_8192_2000.pth')


new_state_dict = OrderedDict()
for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
    name = k[17:] # 截取`module.`后面的xxx.weight
    new_state_dict[name] = v

teacher_model.load_state_dict(new_state_dict, strict=False)

s_state_dict = torch.load('student_model_bagel_t5s401_contr.pth')
student_model.load_state_dict(s_state_dict)

data_path = "../mvtec3d/rope/vis"

dataset = bagle(data_path=data_path)
if len(dataset) == 0:
    print(f"警告: 数据集 {data_path} 为空，跳过处理。")
    
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
n=0
for i, (taxonomy_id, model_id, points) in enumerate(data_loader):
    points = points.to('cuda')
    B, N, _ = points.shape
    distance_matrices = torch.zeros((B, N, N), device='cuda')  # 创建一个空的张量来存储距离矩阵
    
    for b in range(B):
        # 对每个批次中的点云计算距离矩阵
        distance_matrix = torch.cdist(points[b], points[b], p=2)
        # 计算非对角线元素的平均值
        sum_distances = torch.sum(distance_matrix) - torch.sum(torch.diagonal(distance_matrix))
        num_distances = N * (N - 1)  # 非对角线元素的数量
        mean_distance = sum_distances / num_distances
        print(f"Batch {b} mean distance: {mean_distance.item():.4f}")
    neighborhoods, centers, idxs, mas = [], [], [], []
    for i in range(len(student_model.group_dividers)):
        if i == 0:
            neighborhood, center, idx = student_model.group_dividers[i](points)
            ma = neighborhood + center.unsqueeze(2)
        else:
            neighborhood, center, idx = student_model.group_dividers[i](center)
            ma = neighborhood + center.unsqueeze(2)
        
        
        neighborhoods.append(neighborhood)
        centers.append(center)
        idxs.append(idx)  # neighbor indices
        print(f"Layer {i} - Center shape: {center.shape}")
    
    
    first_layer_centers = centers[0].cpu().numpy()
    first_layer_pcd = o3d.geometry.PointCloud()
    first_layer_pcd.points = o3d.utility.Vector3dVector(first_layer_centers.reshape(-1, 3))

    # 识别第一层中心点的边缘
    
    distances = compute_neighbor_distances(first_layer_pcd, k=10)
    edge_threshold = np.percentile(distances, 95)
    edge_indices = np.where(distances > edge_threshold)[0]
    

    # 获取教师模型的输出
    with torch.no_grad():
        teacher_output,_,_ = teacher_model(neighborhoods,centers,idxs,eval=True)
        print(f"Teacher output shape: {teacher_output[1].shape}")  # 检查输出形状
    # 获取学生模型的输出
    student_output,_,_ = student_model(neighborhoods,centers,idxs,eval=True)
    print(f"Student output shape: {student_output[1].shape}")  # 检查输出形状
    
    anomaly_scores = []
    '''
    for student_tensor, teacher_tensor in zip(student_output, teacher_output):
        print("Teacher tensor shape:", teacher_tensor.shape)
        print("Student tensor shape:", student_tensor.shape)
        max_score = 0
        scores = torch.norm(teacher_tensor - student_tensor, dim=2)
        max_score = max(max_score, scores.max().item())
        normalized_anomaly_scores =scores / max_score 
        anomaly_scores.append(normalized_anomaly_scores)
    '''
    student_tensor, teacher_tensor = student_output[0], teacher_output[0]
   
    
    # 计算异常分数
    max_score = 0
    
    
    scores = torch.norm(teacher_tensor - student_tensor,dim=2)
    max_score = max(max_score, scores.max().item())
    normalized_anomaly_scores = scores / max_score
    
    # 将第一层的异常分数添加到列表
    anomaly_scores = [normalized_anomaly_scores]
    #anomaly_scores = [scores]
    anomaly_tensor = anomaly_scores[0]
   
    # 检查 edge_indices 是否在张量的索引范围内
    
    for idx in edge_indices:
        if idx < anomaly_tensor.shape[1]:  # 确保 idx 在张量的第二维范围内
            anomaly_tensor[0, idx] = 0  # 修改对应的异常分数
    
    # 如果需要，可以将修改后的张量放回列表中
    anomaly_scores[0] = anomaly_tensor

    
    
    save_ply(f'rv/anomaly_heatmap_layer_{n}.ply', centers[0], anomaly_tensor.cpu().detach().numpy())
    n=n+1
    '''
    for i, (layer_centers, layer_scores) in enumerate(zip(centers, anomaly_scores)):
        points = np.array([center.cpu().numpy() for center in layer_centers])
        print("Points array shape:", points.shape)

        scores = layer_scores.cpu().detach().numpy()
        flattened_scores = scores.flatten()
        plt.hist(flattened_scores, bins=100, color='blue', alpha=0.7)
        plt.title('Anomaly Scores Distribution')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.show()    
        #plt.savefig('anomaly_scores_distribution.png')
        
    '''



































