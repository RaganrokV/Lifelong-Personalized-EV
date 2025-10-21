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
chunk_size = 64
d_model = 256
feat_dim = 55
max_len = 100
num_layers = 6
nhead = 4
dim_feedforward = 1023
dropout = 0.1

# === 模型初始化 ===
# 初始化 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

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


ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/best_ECM.pt')
# ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/ECM_checkpoint_epoch_100.pt')

epoch_list = [3,5,8,10]       # 
lr_list = [1e-6,2e-6,1e-7,5e-5]         # 不同的在线优化器学习率

results = []  # 用于存储每个超参数组合的评估结果

for epochs in epoch_list:
    for lr in lr_list:
        total_veh=[]
        counter = 0  # 初始化计数器

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

            # 冻结 encoder_pretrained 的参数
            # for param in encoder_pretrained.parameters():
            #     param.requires_grad = False
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
            # epochs=5
            # lr=1e6
            online_optimizer = torch.optim.AdamW(ECM.parameters(), lr=lr, 
                                                 betas=(0.9, 0.999), weight_decay=1e-1)
            online_scheduler = torch.optim.lr_scheduler.StepLR(online_optimizer, 
                                     1, gamma=0.9, last_epoch=-1, verbose=True)

            # online_optimizer = torch.optim.AdamW(energy_head.parameters(), lr=1e-7, 

            preds, reals, old_task_mae = [], [], []

            infer_time = 0.0  # 
            train_time = 0.0 
            a=[]
            for step, (x, y) in enumerate(Dataloaders_test):

                start_infer_time = time.time()  # 开始计时
                #计算每一个chunk的loss

                x, y = x.to(device), y.to(device)

                # == 在每一次任务开始前，模型评估在旧任务上的loss，测试遗忘===
                ECM.eval() 
                with torch.no_grad():
                    pred_y = ECM(x)
                    preds.append(pred_y.cpu().detach())
                    reals.append(y.cpu().detach())
                    old_task_mae.append(evaluation(ECM(trainX.to(device)).detach().cpu().numpy(),
                                                   trainY.numpy())[0])

                end_infer_time = time.time()  # 结束计时

                #  task infer time
                task_infer_time = (end_infer_time - start_infer_time) 

                start_train_time = time.time()  # 开始计时
                # 在线学习更新

                chunk_loss = 0.0
                ECM.train()
                # online_optimizer.zero_grad() #关键性的不同在于不应该调用梯度初始化！
                for epoch in range(epochs):  # 在线更新 
                    # for param in ECM.parameters():
                    #     if not param.requires_grad:
                    #         print(f"Layer {param} requires_grad is set to False.")

                    # online_optimizer.zero_grad()
                    online_pred_y = ECM(x)
                    loss = criterion(online_pred_y, y)

                    # for param in ECM.parameters():
                    #     if param.grad is not None:
                    #         print(f"Grad norm for {param.name}: {param.grad.norm().item()}")

                    loss.backward()
                    online_optimizer.step()
                    chunk_loss += loss.item() 

                    # online_scheduler.step(loss.item())        

                # from sklearn.linear_model import HuberRegressor
                # a.append(chunk_loss)


                print(f"Chunk: {step + 1}/{len(Dataloaders_test)}, chunk_loss: {chunk_loss:.4f}")

                online_scheduler.step()  

                end_train_time = time.time()  # 结束计时

                #  task infer time
                task_train_time = (end_train_time - start_train_time)  


                infer_time += task_infer_time
                train_time += task_train_time

            # del ECM, online_optimizer
            # 转换预测和目标值
            targets = torch.cat(reals, dim=0).numpy()
            predictions = torch.cat(preds, dim=0).numpy()

            EC_True = targets * 60
            EC_Pred = np.abs(predictions) *60

            # x= np.arange(len(a)).reshape(-1, 1)  # 索引
            # y = np.array(a)  # chunk_means 值

            # # model = LinearRegression()
            # model = HuberRegressor(epsilon=1.2)

            # model.fit(x, y)
            # slope = model.coef_[0]  # 直接获取斜率
            # 计算指标
            DirectMeasure = np.array(evaluation(EC_True[1:].reshape(-1,1), 
                                                EC_Pred[1:].reshape(-1,1)))
            FWT, FWT_slope = compute_forward_transfer(EC_True, EC_Pred, dist,chunk_size)
            BWT = compute_backward_transfer(old_task_mae, 60, 0)


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

        results.append({'epoch': epochs+1, 'learning_rate': lr, 'metric': df.mean()})

        
for result in results: print(result)


# %%
# import numpy as np
# import matplotlib.pyplot as plt

# # 创建三种可视化形式
# plt.figure(figsize=(18, 6))

# # 1. 折线对比图
# plt.subplot(1, 3, 1)
# plt.plot(EC_True[1:], label='True Values', color='#2c7bb6', linewidth=2)
# plt.plot(EC_Pred[1:], label='Predicted', color='#d7191c', linestyle='--', alpha=0.8)
# plt.xlabel('Sample Index', fontsize=12)
# plt.ylabel('Energy Consumption', fontsize=12)
# plt.title('True vs Predicted Values Comparison', fontsize=14)
# plt.legend()
# plt.grid(True, alpha=0.3)

# # 2. 散点图与理想线
# plt.subplot(1, 3, 2)
# plt.scatter(EC_True[1:], EC_Pred[1:], 
#             c='#2c7bb6', alpha=0.6, 
#             edgecolors='w', linewidths=0.5)
# plt.plot([min(EC_True), max(EC_True)], [min(EC_True), max(EC_True)], 
#          '--', color='#d7191c', linewidth=1.5)
# plt.xlabel('True Values', fontsize=12)
# plt.ylabel('Predicted Values', fontsize=12)
# plt.title('Prediction Accuracy Analysis', fontsize=14)
# plt.grid(True, alpha=0.3)

# # 3. 残差分布图
# residuals = EC_Pred[1:].reshape(-1,1) - EC_True[1:].reshape(-1,1)
# plt.subplot(1, 3, 3)
# plt.hist(residuals, bins=30, 
#          color='#2c7bb6', 
#          edgecolor='white', 
#          density=True)
# plt.xlabel('Prediction Residuals', fontsize=12)
# plt.ylabel('Density', fontsize=12)
# plt.title('Error Distribution', fontsize=14)
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()
# # %%
# # for name, param in ECM.state_dict().items():
# #     if "weight" in name or "bias" in name:  # 只打印参数（权重和偏置）
# #         print(f"Layer: {name}, Shape: {param.shape}, Mean: {param.float().mean().item():.6f}")

# # %%


# # 加载检查点
# ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/best_ECM.pt')
# # ECM.load_state_dict(ckpt['model_state_dict'])

# # 遍历所有参数并打印统计信息
# print(f"{'Layer Name':<50} | {'Mean':<12} | {'Std':<12} | {'Min':<12} | {'Max':<12}")
# print("-" * 100)
# for name, param in ECM.named_parameters():
#     with torch.no_grad():
#         # 计算统计量
#         mean = param.mean().item()
#         std = param.std(unbiased=True).item()  # 无偏标准差
#         min_val = param.min().item()
#         max_val = param.max().item()
        
#         # 格式化输出（科学计数法）
#         print(f"{name:<50} | "
#               f"{mean:>12.4e} | "
#               f"{std:>12.4e} | "
#               f"{min_val:>12.4e} | "
#               f"{max_val:>12.4e}")
# # %%
# gradients = {n: p.grad for n, p in ECM.named_parameters() if p.requires_grad}
# print("\n=== Gradient Statistics ===")
# for name, grad in gradients.items():
#     if grad is not None:
#         print(f"Layer: {name}")
#         print(f"  Gradient mean: {grad.min().item():.6e}")
#         print(f"  Gradient shape: {grad.shape}")
#         print("-" * 50)
#     else:
#         print(f"Warning: {name} has no gradient!")  
# %%
