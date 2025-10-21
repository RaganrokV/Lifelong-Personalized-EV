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
from models import  monte_carlo_predict,compute_prediction_intervals,compute_picp_mpiw
#%%

class EWC:
    def __init__(self, model, dataloader, criterion, device):
        """
        EWC 初始化
        :param model: 预训练模型
        :param dataloader: 用于计算费舍尔矩阵的旧任务数据加载器
        :param criterion: 损失函数（例如 nn.CrossEntropyLoss）
        :param device: 计算设备（'cuda' 或 'cpu'）
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.model_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        """
        计算费舍尔信息矩阵
        :return: 字典形式存储的费舍尔矩阵
        """
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()  # 确保模型在评估模式
        for inputs, labels in self.dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()  # 反向传播

            for n, p in self.model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[n] += p.grad.detach() ** 2

        # 对费舍尔信息矩阵取平均
        for n in fisher:
            fisher[n] /= (2 * len(self.dataloader))
        return fisher
    
    def update_fisher(self):
        self.fisher = self._compute_fisher()


    def ewc_loss(self, model, lambda_ewc):
        """
        计算 EWC 正则化项
        :param model: 当前模型
        :param lambda_ewc: 正则化系数
        :return: 正则化损失
        """
        ewc_loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                ewc_loss += 0.5 * lambda_ewc * torch.sum(self.fisher[n] * (p - self.model_params[n]) ** 2)
        return lambda_ewc * ewc_loss

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 初始化 GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
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




# lambda_ewc_list=[1e3,5e3,1e4,2e4,5e4,1e5,5e5,1e6]
lambda_ewc_list=[2e4]
results = []  # 用于存储每个超参数组合的评估结果
lambda_ewc=2e4
chunk_size=64

ALL_PICP,ALL_MPIW=[],[]
# for lambda_ewc in lambda_ewc_list:
# for chunk_size in [8,16,32,64,96,128]:
# mc_samples_list=[20,50,100]
mc_samples_list=[0]
for mc_samples in mc_samples_list:
    total_veh=[]
    counter = 0  # 初始化计数器
    mae_evolution=[]
    forget_evolution=[]
    PICP,MPIW=[],[]
    for car_id, ev_data in EV_dict.items():

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



        # np.random.seed(42)
        # torch.manual_seed(42)
        # torch.cuda.manual_seed_all(42)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # ==== 微调阶段 ======
        fine_tune_model(ECM, Dataloaders_train, criterion, optimizer, 
                        scheduler, device,num_epochs = 5)

        del optimizer, scheduler

        #% 在线学习阶段
        epochs=5
        online_optimizer = torch.optim.AdamW(ECM.parameters(), lr=1e-6, 
                                             betas=(0.9, 0.999), weight_decay=1e-1)
        online_scheduler = torch.optim.lr_scheduler.StepLR(online_optimizer, 
                                 1, gamma=0.5, last_epoch=-1, verbose=True) #or gamma=0.9

        # 初始化 EWC
        ewc = EWC(ECM, Dataloaders_train, criterion, device)

        preds, reals, old_task_mae = [], [], []

        chunk_metric=[]
        
        picp_list, mpiw_list =[],[]

        infer_time = 0.0  # 
        train_time = 0.0 
        for step, (x, y) in enumerate(Dataloaders_test):

            start_infer_time = time.time()  # 开始计时
            #计算每一个chunk的loss

            # x, y = x.to(device), y.to(device)

            # # == 在每一次任务开始前，模型评估在旧任务上的loss，测试遗忘===
            # mean_pred, std_pred = monte_carlo_predict(ECM, x, mc_samples=mc_samples)

            # lower_bound, upper_bound = compute_prediction_intervals(mean_pred, std_pred, z=1.96)

            # # 假设 true_values 是你的真实标签，均为 numpy 数组
            # picp, mpiw = compute_picp_mpiw(lower_bound.cpu().numpy(), upper_bound.cpu().numpy(), y.cpu().numpy())

            # picp_list.append(picp)
            # mpiw_list.append(mpiw)
            # print("PICP: {:.2f}, MPIW: {:.2f}".format(picp, mpiw))


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
                loss = criterion(online_pred_y, y) + ewc.ewc_loss(ECM, 
                                                                  lambda_ewc=lambda_ewc)
                loss.backward()
                online_optimizer.step()
                chunk_loss += loss.item()  

                # online_scheduler.step(loss.item())        
            print(f"Chunk: {step + 1}/{len(Dataloaders_test)}, chunk_loss: {chunk_loss:.4f}")

            # online_scheduler.step(chunk_loss)
            online_scheduler.step()

            ewc.update_fisher() 

            end_train_time = time.time()  # 结束计时

            #  task infer time
            task_train_time = (end_train_time - start_train_time)  


            infer_time += task_infer_time
            train_time += task_train_time

        del ECM, ewc, online_optimizer,online_scheduler

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

        PICP.append(np.array(picp_list))
        MPIW.append(np.array(mpiw_list))

        mae_evolution.append(np.array(chunk_metric))

        forget_evolution.append(np.array(old_task_mae))


        print(f"Direct Measure: {DirectMeasure}, FWT: {FWT}, BWT: {BWT}")

        # break

        row = np.concatenate([DirectMeasure, [FWT, FWT_slope, BWT,
                                              infer_time/len(Dataloaders_test.dataset), 
                                              train_time/len(Dataloaders_test.dataset)]])
        total_veh.append(row)

    columns = ["MAE", "RMSE", "MAPE", "SMAPE", "R2", "FWT","FWT_slope", "BWT", "InferenceTime", "TrainingTime"]
    df = pd.DataFrame(total_veh, columns=columns)

    print(df)
    #%
    print(df.mean())

    print((df<0).sum())

    # results.append({'lambda_ewc': lambda_ewc, 'metric': df.mean()})
    results.append({'mc_samples': mc_samples, 'metric': df.mean()})

    ALL_PICP.append(PICP)
    ALL_MPIW.append(MPIW)


for result in results: print(result)

#%%
# import pickle

# ECW_DATA={'metric': df.mean(),'individula_metric': df,
# 'mae_evolution': mae_evolution,'forget_evolution':forget_evolution}

# pickle.dump(ECW_DATA, 
#             open('/home/ps/haichao/1-lifelong_learning/plot_data/ECW_DATA.pkl', 'wb'))
# %%
# import pickle

# ECW_DATA={'MPIW': ALL_MPIW,'PICP': ALL_PICP}

# pickle.dump(ECW_DATA, 
#             open('/home/ps/haichao/1-lifelong_learning/plot_data/ECW_prob.pkl', 'wb'))