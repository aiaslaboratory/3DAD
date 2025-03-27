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


args = SimpleNamespace()
args.config = 'cfgs/pre-training/point-m2ae.yaml'  # 配置文件路径
args.resume = False
args.local_rank = 0
args.experiment_path = 'experiments'  # 根据需要设置实验路径

# 加载配置
config = get_config(args)

# 确保实验路径存在
os.makedirs(args.experiment_path, exist_ok=True)

#数据集设置
dataset = bagle(data_path="../mvtec3d/bagel/traintxt/good")
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

val_dataset = bagle(data_path="../outputs/bagel/valtxt")  
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

teacher_model = H_Encoder(config.model)
student_model = H_Encoder(config.model)
teacher_model = teacher_model.to('cuda')
student_model = student_model.to('cuda')
teacher_model.eval()

'''
state_dict = teacher_model.state_dict()

# 打印状态字典的键
print("模型的状态字典键:")
for key in state_dict.keys():
    print(key)
checkpoint = torch.load('m2ae_bagel_8192_2000.pth')  # 替换为您的参数文件路径

# 获取参数字典的键
state_dict_keys = checkpoint.keys()

# 打印参数文件的键
print("参数文件的键:")
for key in state_dict_keys:
    print(key)

'''



state_dict = torch.load('m2ae_bagel_8192_2000.pth')


new_state_dict = OrderedDict()
for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
    name = k[17:] # 截取`module.`后面的xxx.weight
    new_state_dict[name] = v

#encoder_state_dict = {k: v for k, v in new_state_dict.items() if k.startswith('h_encoder.')}




teacher_model.load_state_dict(new_state_dict, strict=False)
student_state_dict = torch.load('student_model.pth')
student_model.load_state_dict(student_state_dict)
# 均方误差
distillation_loss = nn.MSELoss()
optimizer = AdamW(student_model.parameters(), lr=0.001, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0)

epochs = 100
for epoch in range(epochs):
    student_model.train()
    total_loss = 0

    for i, (taxonomy_id, model_id, points) in enumerate(data_loader):
        # 获取输入数据
        # 假设输入数据已经正确地被加载和预处理
        input_data = points.to('cuda')

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
            teacher_output,_,_ = teacher_model(neighborhoods,centers,idxs,eval=True)

        # 获取学生模型的输出
        student_output,_,_ = student_model(neighborhoods,centers,idxs)
        loss = 0
        average_loss = 0
        # 遍历每一对张量
        '''
       
        for student_tensor, teacher_tensor in zip(student_output, teacher_output):
            # 确保张量的形状相同
            # 例如：student_tensor = student_tensor.view_as(teacher_tensor)
        
            # 计算当前张量对的损失
            loss = distillation_loss(student_tensor, teacher_tensor)
        
            # 累加损失
            average_loss += loss
        
        # 计算平均损失（可选）
        average_loss = average_loss / len(student_output)
        '''
        # 获取第一对教师模型和学生模型的输出张量
        student_tensor = student_output[0]
        teacher_tensor = teacher_output[0]
        '''
        print(student_tensor)
        print(teacher_tensor)
        '''
        # 直接计算第一对输出张量的损失
        loss = distillation_loss(student_tensor, teacher_tensor)
        loss = loss
       
        

        
        loss.backward()
        
        
        
        
        optimizer.step()
    
        # 累计损失
        total_loss += loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    
    # 打印每个epoch的损失
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.10f}")
    
    student_model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (taxonomy_id, model_id, points) in enumerate(val_loader):
            input_data = points.to('cuda')
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

            teacher_output, _, _ = teacher_model(neighborhoods, centers, idxs, eval=True)
            student_output, _, _ = student_model(neighborhoods, centers, idxs)
            val_loss = 1000000 * distillation_loss(student_output[0], teacher_output[0])

            total_val_loss += val_loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Validation Average Loss: {avg_val_loss:.10f}")


torch.save(student_model.state_dict(), "student_model_bagel_t5s501.pth")
































