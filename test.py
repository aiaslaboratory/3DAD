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


print(os.environ['CUDA_VISIBLE_DEVICES'])
# 清理 GPU 缓存
print(torch.cuda.is_available())

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

def generate_log_filename(category):
    """生成针对每个类别的日志文件名"""
    return f"logs/{category}_8192training_loss.log"

def validate(model, data_loader):
        model.eval()# 将模型设置为评估模式
        total_loss = 0
        with torch.no_grad():  # 关闭梯度计算
            for i, (taxonomy_id, model_id, points) in enumerate(data_loader):
                points = points.to('cuda')
                loss, _, _, _, _, _, _, _, _ = model(points)
                loss = loss.mean()
                total_loss += loss.item()
        return total_loss / len(data_loader)


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


mvtec3d_root='../mvtec3d'
categories = os.listdir(mvtec3d_root)


for category in categories:
    
    if category.startswith('.'):
        continue
    category_path = os.path.join(mvtec3d_root, category)
    if not os.path.isdir(category_path):
        continue
    log_file = generate_log_filename(category)
    with open(log_file, "w") as f:
        f.write("Epoch, Loss, Val_loss\n")  # 初始化日志文件头
    #数据集设置
    # 数据集设置
    train_dataset = bagle(data_path=os.path.join(category_path, 'traintxt/good'))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    val_dataset = bagle(data_path=os.path.join(category_path, 'testtxt/good'))  # 假设验证集路径类似
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
       
    
    
    
    model = Point_M2AE(config.model)
    '''
    
    state_dict = torch.load('ckpts/pre-train.pth')
    model_weights = state_dict['base_model']
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
        name = k[7:] # 截取`module.`后面的xxx.weight
        new_state_dict[name] = v
    
    model.load_state_dict(model_weights,strict = False)
    '''
    
    checkpoint = torch.load('ckpts/pre-train.pth')

    model_weights = checkpoint['base_model']
    filtered_state_dict = OrderedDict()
    
    # 获取模型的状态字典
    model_state_dict = model.state_dict()
    
    # 遍历检查点中的条目
    for name, param in model_weights.items():
        # 检查模型中是否存在同名的层
        if name in model_state_dict:
            # 检查形状是否匹配
            if param.size() == model_state_dict[name].size():
                filtered_state_dict[name] = param
            else:
                print(f"Warning: Skipping '{name}' due to size mismatch.")
        else:
            print(f"Warning: Layer '{name}' not found in model.")
    
    # 使用过滤后的状态字典加载权重
    model.load_state_dict(filtered_state_dict, strict=False)





    
    
    
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    
    
    
    model = model.to('cuda')
    
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0)
    
    
    patience = 40
    best_val_loss = float('inf')
    counter = 0
    
    # 训练循环
    max_epochs = 200
    for epoch in range(max_epochs):
        
        
        losses = []
        model.train()
        for i, (taxonomy_id, model_id, points) in enumerate(train_loader):
            # 获取输入数据
            
            points = points.to('cuda')
            
    
            # 前向传播
            loss,_, _, _,_,_,_,_,_= model(points)
            loss = loss.mean()
            
    
            
    
            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            loss = loss.mean().item()
            losses.append(loss)
            optimizer.step()
        
        
        avg_epoch_loss = sum(losses) / len(losses)
        
        val_loss = validate(model, val_loader)
        
        with open(log_file, "a") as f:
                f.write(f"{epoch},{avg_epoch_loss},{val_loss}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # 保存最佳模型
            model_save_path = f"p_m2ae_{category}_8192_newbest.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到（最佳验证）: {model_save_path}")
        else:
            counter += 1
            if counter >= patience:
                print(f"模型在{category}上的训练提前停止，因为验证损失在最近{patience}轮内没有改进。")
                break  # 跳出训练循环
        
        
        
        # 更新学习率
        scheduler.step()
        
        # 打印进度信息
        #if epoch % 5 == 0:
            #print(f"Epoch [{epoch}/{max_epochs}], Loss: {loss.item()}")
    
    # 保存模型
    #model_save_path = f"p_m2ae_{category}_1024_1000.pth"
    #torch.save(model.state_dict(), model_save_path)
    #print(f"模型已保存到: {model_save_path}")
    #print(f"{category}的日志已保存至: {log_file}")




'''

with open("../outputs/bagel/testtxt/crack/000.txt", "r", encoding='utf-8') as f:  #打开文本
    points = []
    for line in f:
        x, y, z = map(float, line.split(' ')[:3])
        points.append([x, y, z])

point= torch.tensor(points).float() 
point = point.unsqueeze(0) 

point = point.to('cuda')
print(next(model.parameters()).device)





#loss,original_points, masked_points, reconstructed_points=model(point)
'''




'''
original_points = original_points.reshape(-1, 3)
masked_points = masked_points.reshape(-1, 3)
reconstructed_points = reconstructed_points.reshape(-1, 3)


save_point_cloud(original_points, 'bagleoriginal.ply')
save_point_cloud(masked_points, 'bagelmasked.ply')
save_point_cloud(reconstructed_points, 'bagelreconstructed.ply')


'''













