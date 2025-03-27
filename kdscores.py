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



def compute_neighbor_distances(pcd, k=5):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distances = []

    for point in pcd.points:
        [_, idx, dist] = pcd_tree.search_knn_vector_3d(point, k)
        distances.append(np.mean(dist))

    return np.array(distances)



args = SimpleNamespace()
args.teacher_config = 'cfgs/pre-training/point-m2ae.yaml'  # 教师模型配置文件路径
args.student_config = 'cfgs/pre-training/point-m2ae.yaml'  # 学生模型配置文件路径
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
teacher_model = Point_M2AE(teacher_config.model)
student_model = Point_M2AE(student_config.model)
teacher_model = teacher_model.to('cuda')
student_model = student_model.to('cuda')
teacher_model.eval()
student_model.eval()
state_dict = torch.load('p_m2ae_carrot_8192_best.pth')


new_state_dict = OrderedDict()
for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
    name = k[7:] # 截取`module.`后面的xxx.weight
    new_state_dict[name] = v


student_state_dict = torch.load('student_model_carrot_8192ae55.pth')
#encoder_state_dict = {k: v for k, v in new_state_dict.items() if k.startswith('h_encoder.')}

teacher_model.load_state_dict(new_state_dict, strict=False)
student_model.load_state_dict(student_state_dict)


test_folder_path = "../mvtec3d/carrot/testtxt"
output_path = "txt" 
class_folders = [f for f in os.listdir(test_folder_path) if os.path.isdir(os.path.join(test_folder_path, f))]


torch.cuda.empty_cache()

for folder in class_folders:
    torch.cuda.empty_cache()
    folder_path = os.path.join(test_folder_path, folder)
    scores_path = os.path.join(output_path, f"{folder}_anomaly_scores.txt")
    dataset = bagle(data_path=folder_path)
    if len(dataset) == 0:
        print(f"警告: 数据集 {folder_path} 为空，跳过处理。")
        continue
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # 不打乱顺序，以保持文件名对应
    
    with open(scores_path, "w") as scores_file:
        for i, (taxonomy_id, model_id, points) in enumerate(data_loader):
           



            torch.cuda.empty_cache()
            points = points.to('cuda')
        
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
            
            '''
            first_layer_centers = centers[0].cpu().numpy()
            first_layer_pcd = o3d.geometry.PointCloud()
            first_layer_pcd.points = o3d.utility.Vector3dVector(first_layer_centers.reshape(-1, 3))
        
            # 识别第一层中心点的边缘
            
            distances = compute_neighbor_distances(first_layer_pcd, k=10)
            edge_threshold = np.percentile(distances, 90)
            edge_indices = np.where(distances > edge_threshold)[0]
            '''
            # 获取教师模型的输出
            with torch.no_grad():
                teacher_features  = teacher_model(points,eval=True)
        
                # 获取学生模型的输出
                student_features = student_model(points)
            del neighborhoods
            del centers
            del idxs
            
            
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
        
        
        
        
            student_tensor, teacher_tensor = student_features[0], teacher_features[0]
       
            # 计算异常分数
            max_score = 0
            scores = torch.norm(teacher_tensor - student_tensor, dim=1)
            max_score = scores.max().item()
            normalized_scores = scores / max_score
            '''
            # 检查 edge_indices 是否在张量的索引范围内
            for idx in edge_indices:
                if idx < normalized_scores.shape[1]:
                    normalized_scores[0, idx] = 0
            # 将第一层的异常分数添加到列表
           
            '''
            
            
            
            '''
            
           
            threshold = 0.8

            # 筛选出大于阈值的分数
            scores_above_threshold = normalized_scores[scores > threshold]
            
            # 计算这些分数的平均值
            
            average_score = scores_above_threshold.mean()
            
            scores_file.write(f"{average_score}\n")

            '''
            # 选择最高1%的分数
            
            
            top_k_scores = torch.topk(scores, k=int(0.03 * scores.numel()), largest=True).values
            top_k_scores,_ = torch.max(scores, dim=0)
            average_score = top_k_scores.mean().item()
            
            scores_file.write(f"{average_score}\n")

            #student_output = [tensor.detach().cpu() for tensor in student_output]
            #teacher_output = [tensor.detach().cpu() for tensor in teacher_output]

            del teacher_features
            del student_features
            del scores
            del top_k_scores
            torch.cuda.empty_cache()
            
































