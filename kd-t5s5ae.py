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


def generate_log_filename(category):
    """生成针对每个类别的日志文件名"""
    return f"logs/{category} con.log"


def top_k_contrastive_loss(student_features, teacher_features, k_percent=0.03):
    # 计算差异
    differences = (student_features - teacher_features).pow(2)
    num_features = differences.numel()  # 获取特征总数
    k = max(1, int(num_features * k_percent))  # 至少选择一个特征
    
    # 只考虑差异最大的前K%的特征
    top_k_diffs, _ = torch.topk(differences.reshape(-1), k)
    
    # 计算这些点的平均损失
    return top_k_diffs.mean()



args = SimpleNamespace()
args.teacher_config = 'cfgs/pre-training/point-m2ae.yaml'  # 教师模型配置文件路径
args.student_config = 'cfgs/pre-training/point-m2ae.yaml'  # 学生模型配置文件路径
args.resume = False  # 保留你原有的其他参数设置
args.local_rank = 0
args.experiment_path = 'experiments'  # 根据需要设置实验路径

# 加载教师模型配置
teacher_args = SimpleNamespace(**vars(args))  # 创建一个新的args副本
teacher_args.config = args.teacher_config  # 更新配置文件路径 b                                
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


base_path = '../mvtec3d'
teacher_prefix = 'p_m2ae_'
student_prefix = 'student_model_'
patience = 50  # 早停的耐心值，即连续多少轮没有改善则停止
best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
counter = 0  # 初始化早停计数器


# 获取所有类别
categories = os.listdir(os.path.join(base_path))
categories = [cat for cat in categories if os.path.isdir(os.path.join(base_path, cat)) and not cat.startswith('.')]


#for category in categories:
category = 'carrot'
print(f"Training for category: {category}")

train_data_path = os.path.join(base_path, category, 'traintxt', 'good')
val_data_path = os.path.join(base_path, category, 'testtxt', 'good')
teacher_weight_path = os.path.join(teacher_prefix + category + '_8192_best.pth')
student_weight_path = os.path.join(student_prefix + category + '_8192aecon55n1.pth')
student_save_path = student_prefix + category + '_8192ae55.pth'

log_file = generate_log_filename(category)
with open(log_file, "w") as f:
    f.write("Epoch, Loss,Val_loss\n")  # 初始化日志文件头

#数据集设置
dataset = bagle(data_path=train_data_path)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

val_dataset = bagle(data_path=val_data_path)  
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



#encoder_state_dict = {k: v for k, v in new_state_dict.items() if k.startswith('h_encoder.')}
state_dict = torch.load(teacher_weight_path)
new_state_dict = OrderedDict()
for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
    name = k[7:] # 截取`module.`后面的xxx.weight
    new_state_dict[name] = v



teacher_model.load_state_dict(new_state_dict,strict = False)
'''
student_state_dict = torch.load('p_m2ae_carrot_8192_nbest.pth')
s_state_dict = OrderedDict()
for k, v in student_state_dict.items(): # k为module.xxx.weight, v为权重
    name = k[17:] # 截取`module.`后面的xxx.weight
    s_state_dict[name] = v


student_model.load_state_dict(s_state_dict,strict=False)


state_dict = torch.load('ckpts/pre-train.pth')
model_weights = state_dict['base_model']



student_model.load_state_dict(model_weights, strict=False)



student_state_dict = torch.load(student_weight_path)
student_model.load_state_dict(student_state_dict)
'''
# 均方误差
distillation_loss = nn.MSELoss()
optimizer = AdamW(student_model.parameters(), lr=0.001, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0)
best_val_loss_per_cat = float('inf')
patience_per_cat = 200
counter_per_cat = 0
epochs = 200
for epoch in range(epochs):
    student_model.train()
    total_loss = 0
    total_mse_loss = 0
    total_contr_loss = 0
    for i, (taxonomy_id, model_id, points) in enumerate(data_loader):


        
        input_data = points.to('cuda')
        points = points.to('cuda')
        neighborhoods, centers, idxs, mas = [], [], [], []
        for i in range(len(student_model.group_dividers)):
            if i == 0:
                neighborhood, center, idx = student_model.group_dividers[i](input_data)
                ma = neighborhood + center.unsqueeze(2)
            else:
                neighborhood, center, idx = student_model.group_dividers[i](center)
                ma = neighborhood + center.unsqueeze(2)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices
            
        
        
        
        
        
        
        
        # 重置梯度
        
        optimizer.zero_grad()

        # 获取教师模型的输出
        with torch.no_grad():
            teacher_features  = teacher_model(points,eval=True)

        # 获取学生模型的输出
        student_features = student_model(points)
        loss = 0
        average_loss = 0
        # 遍历每一对张量
        
       
        for student_tensor, teacher_tensor in zip(teacher_features, student_features):
            # 确保张量的形状相同
            # 例如：student_tensor = student_tensor.view_as(teacher_tensor)
        
            # 计算当前张量对的损失
            loss = distillation_loss(student_tensor, teacher_tensor)
        
            # 累加损失
            average_loss += loss
        
        # 计算平均损失（可选）
        average_loss = average_loss / len(student_features)
        
        # 获取第一对教师模型和学生模型的输出张量
        student_tensor = student_features[0]
        teacher_tensor = teacher_features[0]
        '''
        print(student_tensor)
        print(teacher_tensor)
        '''
        # 直接计算第一对输出张量的损失
        mse_loss = distillation_loss(student_tensor, teacher_tensor)
        contr_loss = top_k_contrastive_loss(student_tensor, teacher_tensor)
        loss =  mse_loss
        #loss =contr_loss
        

        
        loss.backward()
        
        
        
        
        optimizer.step()
    
        # 累计损失
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_contr_loss += contr_loss.item()
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    avg_mse_loss = total_mse_loss / len(data_loader)
    avg_contr_loss = total_contr_loss / len(data_loader)
    # 打印每个epoch的损失

    
    student_model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (taxonomy_id, model_id, points) in enumerate(val_loader):
            input_data = points.to('cuda')
            points = points.to('cuda')
            neighborhoods, centers, idxs, mas = [], [], [], []
            for i in range(len(student_model.group_dividers)):
                if i == 0:
                    neighborhood, center, idx = student_model.group_dividers[i](input_data)
                    ma = neighborhood + center.unsqueeze(2)
                else:
                    neighborhood, center, idx = student_model.group_dividers[i](center)
                    ma = neighborhood + center.unsqueeze(2)
                neighborhoods.append(neighborhood)
                centers.append(center)
                idxs.append(idx)

            teacher_features  = teacher_model(points,eval=True)
            tudent_features = student_model(points)
            
            
            
            
            
            
            for student_tensor, teacher_tensor in zip(student_features, teacher_features):
            # 确保张量的形状相同
            # 例如：student_tensor = student_tensor.view_as(teacher_tensor)
            
                # 计算当前张量对的损失
                loss = distillation_loss(student_tensor, teacher_tensor)
            
                # 累加损失
                average_loss += loss
            
            # 计算平均损失（可选）
            average_loss = average_loss / len(student_features)
            student_tensor = student_features[0]
            teacher_tensor = teacher_features[0]
            
            
            val_loss = top_k_contrastive_loss(student_tensor, teacher_tensor) 

            total_val_loss += average_loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)


    
    print(f"Epoch [{epoch+1}/{epochs}], Validation Average Loss: {avg_val_loss:.10f}")
    
    
    with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_loss},{avg_val_loss}\n")
    
    
    if avg_val_loss < best_val_loss_per_cat:
            best_val_loss_per_cat = avg_val_loss
            torch.save(student_model.state_dict(), student_save_path)
            counter_per_cat = 0  # 重置计数器
    else:
        counter_per_cat += 1
        print(f"No improvement for {counter_per_cat} epochs for category {category}.")

        # 达到该类别的耐心值后停止训练
        if counter_per_cat >= patience_per_cat:
            print(f"Early stopping triggered for category {category}.")
            break


#torch.save(student_model.state_dict(), student_save_path)































