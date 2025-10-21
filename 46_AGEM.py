#%% -*- coding: utf-8 -*-
from collections import deque
from random import sample
import numpy as np
import pandas as pd
import warnings
from My_utils.evaluation_scheme import evaluation
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import time
import random
from functions import labeling, preprocess_data,normalize
from models import TransformerEncoder,EnergyPredictionHead,GeneralPredictionModel
from models import fine_tune_model,compute_forward_transfer,compute_backward_transfer
from models import  create_dataloaders
from models import  monte_carlo_predict,compute_prediction_intervals,compute_picp_mpiw
#%%
class AGEM:
    def __init__(self, buffer_size, sample_size):
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.buffer = deque(maxlen=buffer_size)

    def add_to_buffer(self, x, y):
        for i in range(len(x)):
            self.buffer.append((x[i].detach().cpu(), y[i].detach().cpu()))

    def sample_from_buffer(self):
        if len(self.buffer) == 0:
            return []
        sample_size = min(len(self.buffer), self.sample_size)
        return random.sample(list(self.buffer), sample_size)
    
    def collect_gradients(self,model):
        """
        收集模型所有参数的梯度，返回一个字典，
        key 为参数名称，value 为梯度张量的克隆。
        """
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_dict[name] = param.grad.clone()
        return grad_dict

    def project_gradient_dict(self,current_grad_dict, ref_grad_dict):
        """
        对每个参数的梯度进行投影，使得
        如果当前梯度与参考梯度的点积为负，则
        当前梯度被投影到参考梯度的方向上。

        参数：
          current_grad_dict: dict, 当前任务梯度（key: parameter name, value: tensor）
          ref_grad_dict: dict, 参考任务梯度（同上）

        返回：
          projected_grad_dict: dict, 投影后的梯度字典
        """
        projected_grad_dict = {}
        for name in current_grad_dict:
            g_cur = current_grad_dict[name]
            if name in ref_grad_dict:
                g_ref = ref_grad_dict[name]
                # 展平当前和参考梯度
                g_cur_flat = g_cur.view(-1)
                g_ref_flat = g_ref.view(-1)
                dot_product = torch.dot(g_cur_flat, g_ref_flat)
                norm_ref = torch.norm(g_ref_flat) ** 2 + 1e-8  # 防止除以0
                if dot_product < 0:
                    g_proj_flat = g_cur_flat - (dot_product / norm_ref) * g_ref_flat
                else:
                    g_proj_flat = g_cur_flat.clone()
                # 将投影后的梯度还原到原始形状
                projected_grad_dict[name] = g_proj_flat.view_as(g_cur)
            else:
                projected_grad_dict[name] = g_cur.clone()
        return projected_grad_dict

    

    def project_gradient(self, current_grads, ref_grads):
        current_grads_flat = torch.cat([g.view(-1) for g in current_grads if g is not None])
        ref_grads_flat = torch.cat([g.view(-1) for g in ref_grads if g is not None])

        if len(ref_grads_flat) == 0:
            return  # 无参考梯度可投影

        dot_product = torch.dot(current_grads_flat, ref_grads_flat)
        norm_ref_grads = torch.norm(ref_grads_flat) ** 2

        if norm_ref_grads <= 1e-10:  # 避免除以零
            return

        if dot_product < 0:
            current_grads_flat.sub_((dot_product / norm_ref_grads) * ref_grads_flat)

        # 将投影后的梯度还原到current_grads
        index = 0
        for g in current_grads:
            if g is not None:
                num_elements = g.numel()
                g.data.copy_(current_grads_flat[index:index+num_elements].view_as(g))
                index += num_elements

#%% 数据预处理和初始化
all_df = pd.read_pickle('/home/ps/haichao/1-lifelong_learning/trip data/all_df.pkl')

labeled_df=labeling(all_df)

df = preprocess_data(labeled_df)

#  === training and testing ===

non_hybrid_data = df[df['Powertrain Type'] == 1]

trip_counts_non_hybrid = non_hybrid_data['car_id'].value_counts()

cars_with_many_trips = trip_counts_non_hybrid[trip_counts_non_hybrid > 1000].index

random.seed(2)
test_car_ids = random.sample(list(cars_with_many_trips), 100)

test_set = df[df['car_id'].isin(test_car_ids)]


test_set = normalize(test_set)
# =====  拆分车辆 =======
EV_dict = {}

for car_id, group in test_set.groupby('car_id'):
    # 删除每个子 DataFrame 中的 'car_id' 列
    group = group.drop(columns=['car_id'])
    EV_dict[car_id] = group


#%%
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#  === 超参数设置 ===
# chunk_size = 64
d_model = 256
feat_dim = 55
max_len = 100
num_layers = 6
nhead = 4
dim_feedforward = 1023
# dropout = 0.1
dropout = 0.5

# === 模型初始化 ===
Attribution_ids = torch.tensor([5] * 9 + [4] * 6 + [3] * 16 + [2] * 8 + [1] * 6 + [0] * 10)
min_val = Attribution_ids.min()
max_val = Attribution_ids.max()
normalized_attribution_ids = (Attribution_ids - min_val) / (max_val - min_val)


encoder_pretrained = TransformerEncoder(normalized_attribution_ids.to(device), d_model=d_model, 
                                        max_len=max_len, num_layers=num_layers, nhead=nhead, 
                                        dim_feedforward=dim_feedforward, dropout=dropout)

# 解码头
energy_head = EnergyPredictionHead(d_model=d_model, feat_dim=feat_dim, 
                                   dropout=dropout).to(device)

# ===== 创建能耗预测模型 =======
ECM = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)


ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/best_ECM.pt')
# ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/ECM_checkpoint_epoch_100.pt')


ALL_PICP,ALL_MPIW=[],[]
# buffer_list=[8,16,32,48,64,100]
buffer_list=[64]
sample_size=64
results = []  # 用于存储每个超参数组合的评估结果
# for chunk_size in [8,16,32,64,96,128]:
# for sample_size in buffer_list:
chunk_size=64
mc_samples_list=[20,50,100]
for mc_samples in mc_samples_list:
    total_veh=[]
    counter = 0  # 初始化计数器
    mae_evolution=[]
    forget_evolution=[]
    PICP,MPIW=[],[]
    for car_id, ev_data in EV_dict.items():
        # 数据预处理和划分
        # 数据预处理和划分
        # if counter >= 5:  # 判断计数器是否已达到5
        #     break  # 退出循环

        # counter += 1  # 每次循环后计数器加1   

        # === 构造数据 ===
        prepared_data = create_dataloaders(ev_data,chunk_size=chunk_size, 
                                           split_ratio=0.1)
        Dataloaders_train=prepared_data[0]
        Dataloaders_test=prepared_data[1]
        dist=prepared_data[2]
        trainX=prepared_data[3]
        trainY=prepared_data[4]

        #  ====== 加载参数 ======
        ECM = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
        ECM.load_state_dict(ckpt['model_state_dict'])


        ECM.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(ECM.parameters(), lr=2e-4, 
                                      betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode='min', patience=1, factor=0.5, verbose=True)

        # ==== 微调阶段 ======
        fine_tune_model(ECM, Dataloaders_train, criterion, optimizer, 
                        scheduler, device,num_epochs = 5)

        del optimizer, scheduler

        #% 在线学习阶段
        epochs=5
        online_optimizer = torch.optim.AdamW(ECM.parameters(), lr=1e-6, 
                                             betas=(0.9, 0.999), weight_decay=1e-1)
        online_scheduler = torch.optim.lr_scheduler.StepLR(online_optimizer, 
                                 1, gamma=0.5, last_epoch=-1, verbose=True)

        # 初始化 AGEM
        buffer_size = 100  # 缓冲区容量
        sample_size = sample_size    # 每次采样大小
        agem = AGEM(buffer_size, sample_size)



        preds, reals, old_task_mae = [], [], []

        picp_list, mpiw_list =[],[]
        chunk_metric=[]
        infer_time = 0.0  # 
        train_time = 0.0 
        for step, (x, y) in enumerate(Dataloaders_test):

            start_infer_time = time.time()  # 开始计时
            #计算每一个chunk的loss

            x, y = x.to(device), y.to(device)

            # == 在每一次任务开始前，模型评估在旧任务上的loss，测试遗忘===
            mean_pred, std_pred = monte_carlo_predict(ECM, x, mc_samples=mc_samples)

            lower_bound, upper_bound = compute_prediction_intervals(mean_pred, std_pred, z=1.96)

            # 假设 true_values 是你的真实标签，均为 numpy 数组
            picp, mpiw = compute_picp_mpiw(lower_bound.cpu().numpy(), upper_bound.cpu().numpy(), y.cpu().numpy())

            picp_list.append(picp)
            mpiw_list.append(mpiw)
            print("PICP: {:.2f}, MPIW: {:.2f}".format(picp, mpiw))
            # ECM.eval() 
            with torch.no_grad():
                pred_y = ECM(x)
                preds.append(pred_y.cpu().detach())
                reals.append(y.cpu().detach())
                old_task_mae.append(evaluation(ECM(trainX.to(device)).detach().cpu().numpy(),
                                               trainY.numpy())[0])
                chunk_metric.append(criterion(pred_y, y).detach().cpu().numpy())

            end_infer_time = time.time()  # 结束计时

            #  task infer time
            task_infer_time = (end_infer_time - start_infer_time) 

            start_train_time = time.time()  # 开始计时

            # 当前任务的前向传播与训练
            chunk_loss = 0.0
            ECM.train()
            for epoch in range(epochs):

                # online_optimizer.zero_grad()
                online_pred_y =  ECM(x)

                supervised_loss = criterion(online_pred_y, y.unsqueeze(1))
                supervised_loss.backward()

                # 获取当前任务的梯度
                current_grad_dict = agem.collect_gradients(ECM)

                # 如果缓冲区有数据，执行梯度投影
                buffer_samples = agem.sample_from_buffer()
                if len(buffer_samples) > 0:
                    buffer_x, buffer_y = zip(*buffer_samples)
                    buffer_x = torch.stack(buffer_x).to(device)
                    buffer_y = torch.stack(buffer_y).to(device)

                    # 计算缓冲区样本的梯度
                    # online_optimizer.zero_grad()
                    buffer_pred = ECM(buffer_x)
                    buffer_loss = criterion(buffer_pred, buffer_y.unsqueeze(1))
                    buffer_loss.backward()
                    ref_grad_dict = agem.collect_gradients(ECM)

                    # 投影当前任务的梯度
                    projected_grad_dict = agem.project_gradient_dict(current_grad_dict, ref_grad_dict)

                    # 将投影后的梯度赋值回模型参数
                    for name, param in ECM.named_parameters():
                            if param.grad is not None and name in projected_grad_dict:
                                param.grad.copy_(projected_grad_dict[name])

                # 更新模型参数
                online_optimizer.step()
                chunk_loss += supervised_loss.item()

            print(f"Chunk: {step + 1}/{len(Dataloaders_test)}, Epoch Loss: {chunk_loss:.4f}")

            online_scheduler.step()

            # 此时成为历史数据，将其加入缓冲区
            agem.add_to_buffer(x, y)

            end_train_time = time.time()  # 结束计时

            #  task infer time
            task_train_time = (end_train_time - start_train_time)  


            infer_time += task_infer_time
            train_time += task_train_time

        # del ECM, ewc, online_optimizer

        # 转换预测和目标值
        targets = torch.cat(reals, dim=0).numpy()
        predictions = torch.cat(preds, dim=0).numpy()

        EC_True = targets * 60
        EC_Pred = np.abs(predictions) *60

        # 计算指标
        DirectMeasure = np.array(evaluation(EC_True[1:].reshape(-1,1), 
                                            EC_Pred[1:].reshape(-1,1)))
        FWT, FWT_slope = compute_forward_transfer(EC_True, EC_Pred, dist,chunk_size)
        BWT = compute_backward_transfer(old_task_mae, 60, 0)

        mae_evolution.append(np.array(chunk_metric))
        forget_evolution.append(np.array(old_task_mae))
        PICP.append(np.array(picp_list))
        MPIW.append(np.array(mpiw_list))


        print(f"Direct Measure: {DirectMeasure}, FWT: {FWT}, BWT: {BWT}")

        # break

        row = np.concatenate([DirectMeasure, [FWT, FWT_slope, BWT,
                                              infer_time/len(Dataloaders_test.dataset), 
                                              train_time/len(Dataloaders_test.dataset)]])
        total_veh.append(row)

    columns = ["MAE", "RMSE", "MAPE", "SMAPE", "R2", "FWT","FWT_slope", "BWT", "InferenceTime", "TrainingTime"]
    df = pd.DataFrame(total_veh, columns=columns)

    print(df)
    #%%
    print(df.mean())

    print((df<0).sum())

    # results.append({'sample_size': sample_size, 'metric': df.mean()})
    results.append({'mc_samples': mc_samples, 'metric': df.mean()})
    ALL_PICP.append(PICP)
    ALL_MPIW.append(MPIW)



for result in results: print(result)


#%%
# import pickle

# AGEM_DATA={'metric': df.mean(),'individula_metric': df,
# 'mae_evolution': mae_evolution,'forget_evolution':forget_evolution}

# pickle.dump(AGEM_DATA, 
#             open('/home/ps/haichao/1-lifelong_learning/plot_data/AGEM_DATA.pkl', 'wb'))

import pickle

AGEM_DATA={'MPIW': ALL_MPIW,'PICP': ALL_PICP}

pickle.dump(AGEM_DATA, 
            open('/home/ps/haichao/1-lifelong_learning/plot_data/AGEM_prob.pkl', 'wb'))
