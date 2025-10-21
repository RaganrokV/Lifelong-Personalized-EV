#%% -*- coding: utf-8 -*-
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
import torch.nn.functional as F
from models import  monte_carlo_predict,compute_prediction_intervals,compute_picp_mpiw

#%%
class LwF:
    def __init__(self, device):
        """
        LwF 初始化
        """
        self.device = device
        self.old_model = None

    def update_old_model(self, model):
        """
        更新旧模型
        :param model: 新模型的拷贝作为旧模型
        """
        self.old_model = ECM
        self.old_model.load_state_dict(model.state_dict())  # 拷贝当前模型参数
        self.old_model.to(self.device)
        self.old_model.eval()  # 冻结旧模型

    def lwf_loss(self, current_outputs, inputs , alpha, temperature):
        """
        计算知识蒸馏损失
        :param current_outputs: 新模型输出
        :param inputs: 输入数据
        :param temperature: 蒸馏温度
        :param alpha: 蒸馏损失权重 
        """
        if self.old_model is None:
            return torch.zeros(1, device=self.device, dtype=torch.float32, requires_grad=True)
        
        with torch.no_grad():
            old_outputs = self.old_model(inputs)
        kd_loss = F.mse_loss(
            F.softmax(current_outputs / temperature, dim=0),
            F.softmax(old_outputs / temperature, dim=0)
            # current_outputs / temperature,old_outputs / temperature
        )
        return alpha * kd_loss
    
#%%
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
# lamda_list=[1e4,1e5,1e6]
# temperature_list=[1,2]
lamda_list=[1e5]
temperature_list=[2]
results = []  # 用于存储每个超参数组合的评估结果
lamda=1e5
temperature=2
chunk_size=64
# for chunk_size in [8,16,32,64,96,128]:
# for temperature in temperature_list:
#     for lamda in lamda_list:
mc_samples_list=[20,50,100]
for mc_samples in mc_samples_list:
        total_veh=[]
        counter = 0  # 初始化计数器
        mae_evolution=[]
        forget_evolution=[]
        PICP,MPIW=[],[]
        for car_id, ev_data in EV_dict.items():
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

            lwf = LwF(device)
            lwf.update_old_model(ECM) 

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
                # 在线学习更新

                chunk_loss = 0.0
                ECM.train()
                # online_optimizer.zero_grad()
                for epoch in range(epochs):  # 在线更新 

                    # online_optimizer.zero_grad()
                    online_pred_y = ECM(x)
                    #区别在与criterion(pred_y, y)的模型在epoch更新参数，而lwf.lwf_loss(pred_y, x)冻结
                    loss = criterion(online_pred_y, y) + lwf.lwf_loss(online_pred_y, x, alpha=lamda,temperature=temperature)

                    print(f"loss: {criterion(online_pred_y, y)}, lwf_loss: {lwf.lwf_loss(online_pred_y, x, alpha= lamda,temperature=temperature)}")

                    loss.backward()
                    online_optimizer.step()
                    chunk_loss += loss.item()  

                    # online_scheduler.step(loss.item())        
                print(f"Chunk: {step + 1}/{len(Dataloaders_test)}, chunk_loss: {chunk_loss:.4f}")

                # online_scheduler.step(chunk_loss)
                online_scheduler.step()
                # 更新旧模型（）强调是否对原始知识的保留
                # lwf.update_old_model(ECM)  

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

        # results.append({'temperature': temperature,'lamda': lamda, 'metric': df.mean()})
        results.append({'mc_samples': mc_samples, 'metric': df.mean()})
        ALL_PICP.append(PICP)
        ALL_MPIW.append(MPIW)

for result in results: print(result)

# import pickle

# LWF_DATA={'metric': df.mean(),'individula_metric': df,
# 'mae_evolution': mae_evolution,'forget_evolution':forget_evolution}

# pickle.dump(LWF_DATA, 
#             open('/home/ps/haichao/1-lifelong_learning/plot_data/LWF_DATA.pkl', 'wb'))

import pickle

LWF_DATA={'MPIW': ALL_MPIW,'PICP': ALL_PICP}

pickle.dump(LWF_DATA, 
            open('/home/ps/haichao/1-lifelong_learning/plot_data/LWF_prob.pkl', 'wb'))