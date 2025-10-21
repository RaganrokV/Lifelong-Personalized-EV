#%% -*- coding: utf-8 -*-
from re import L
import numpy as np
import pandas as pd
import warnings
import pickle
import math
from My_utils.evaluation_scheme import evaluation
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import time
import torch.utils.data as Data
import torch.nn.functional as F
import random
#%%
ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/best_ECM.pt')
state_dict = ckpt["model_state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_key = key[7:]  # 去除 'module.' 的前缀
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

all_df = pd.read_pickle('/home/ps/haichao/1-lifelong_learning/trip data/all_df.pkl')

#%%
"""encoder 分类变量"""
def labeling(df):
    
    # 对出行季节进行Label Encoding
    season_mapping = {'spring': 1, 'summer': 2, 'autumn': 3, 'winter': 4}
    df['当前季节'] = df['当前季节'].map(season_mapping).astype(int)

    # 对出行时段进行Label Encoding
    period_mapping = {'morning peak': 1, 'night peak': 2, 'other time': 3, "night time": 4}
    df['当前时段'] = df['当前时段'].map(period_mapping).astype(int)

    week_mapping = {'weekday': 0, 'weekend': 1}
    df['当前是否工作日'] = df['当前是否工作日'].map(week_mapping).astype(int)

    # 对车辆类型进行Label Encoding
    vehicle_mapping = {'Sedan': 1, 'SUV': 2,'客车-公交': 3,'物流': 4}
    df['车型'] = df['车型'].map(vehicle_mapping).fillna(5).astype(int)

    # 对电池类型进行Label Encoding
    battery_mapping = {'三元材料电池': 1, '磷酸铁锂电池': 2}
    df['电池类型'] = df['电池类型'].map(battery_mapping).fillna(3).astype(int)

    # 对动力类型进行Label Encoding
    power_mapping = {'BEV': 1, 'PHEV': 2}
    df['动力类型'] = df['动力类型'].map(power_mapping).fillna(3).astype(int)

    return df

labeled_df=labeling(all_df)

"替换为英文名"

df = labeled_df.rename(columns={
    '行程能耗': 'Energy Consumption',
     # trip feature
    '行程时间': 'Trip Duration',
    '行程距离': 'Trip Distance',
    '行程平均速度(mean)': 'Avg. Speed',
    '当前月份': 'Month',
    '当前几点': 'Hour',
    '当前星期几': 'Day of the Week',
    '当前季节': 'Season',
    '当前时段': 'Time Period',
    '当前是否工作日': 'Is Workday',
    # battery feature
    '当前SOC': 'State of Charge',
    '当前累积行驶里程': 'Accumulated Driving Range',
    '当前单体电池电压极差': 'Voltage Range',
    '当前单体电池温度极差': 'Temperature Range',
    '当前绝缘电阻值': 'Insulation',
    '累计平均能耗': 'Historic Avg. EC',
    # driving feature
    '前三个行程能量回收比例': 'Energy Recovery Ratio',
    '前三个行程加速踏板均值': 'Avg. of Acceleration Pedal',
    '前三个行程加速踏板最大值': 'Max of Acceleration Pedal',
    '前三个行程加速踏板标准差': 'Std. of Acceleration Pedal',
    '前三个行程制动踏板均值': 'Avg. of Brake Pedal',
    '前三个行程制动踏板最大值': 'Max of Brake Pedal',
    '前三个行程制动踏板标准差': 'Std. of Brake Pedal',
    '前三个行程瞬时速度均值': 'Avg. of Instantaneous Speed',
    '前三个行程瞬时速度最大值': 'Max of Instantaneous Speed',
    '前三个行程瞬时速度标准差': 'Std. of Instantaneous Speed',
    '前三个行程加速度均值': 'Avg. of Acceleration',
    '前三个行程加速度最大值': 'Max of Acceleration',
    '前三个行程加速度标准差': 'Std. of Acceleration',
    '前三个行程减速度均值': 'Avg. of Deceleration',
    '前三个行程减速度最大值': 'Max of Deceleration',
    '前三个行程减速度标准差': 'Std. of Deceleration',
    # charging feature
    '累计等效充电次数': 'Equivalent Recharge Count',
    '平均充电时长': 'Avg. Recharge Duration',
    '累计充电时长': 'Cumulative Recharge Duration',
    '最大充电时长': 'Max Recharge Duration',
    '最小充电时长': 'Min Recharge Duration',
    '起始SOC均值': 'Avg. Starting SOC',
    '截止SOC均值': 'Avg. Ending SOC',
    '充电SOC均值': 'Avg. Recharge SOC',
    # environmental feature
    '温度': 'Temperature',
    '气压mmHg': 'Air Pressure',
    '相对湿度': 'Relative Humidity',
    '风速m/s': 'Wind Speed ',
    '能见度km': 'Visibility',
    '前6h降水量mm': 'Avg. Precipitation',
    # vehicle feature
    '满载质量': 'Gross Vehicle Weight',
    '整备质量': 'Curb Weight',
    '车型': 'Vehicle Model',
    '电池类型': 'Battery Type',
    '动力类型': 'Powertrain Type',
    '电池额定能量': 'Battery Rated Power',
    '电池额定容量': 'Battery Rated Capacity',
    '最大功率': 'Max Power',
    '最大扭矩': 'Max Torque',
    '官方百公里能耗': 'Official ECR',
})

df['Gross Vehicle Weight'] = pd.to_numeric(df['Gross Vehicle Weight'], errors='coerce')
df['Max Power'] = pd.to_numeric(df['Max Power'], errors='coerce')
df['Max Torque'] = pd.to_numeric(df['Max Torque'], errors='coerce')
df['Official ECR'] = pd.to_numeric(df['Official ECR'], errors='coerce')
df['Battery Rated Capacity'] = pd.to_numeric(df['Battery Rated Capacity'], errors='coerce')
# df = df.drop('car_id', axis=1)

df.loc[:, df.columns != 'car_id'] = df.loc[:, df.columns != 'car_id'].apply(
    lambda x: x.fillna(x.median()), axis=0)  #均值和0填补，均可


#%%
max_min_dict = {
    # Trip Features  绝大部分取99%上界
    'Energy Consumption': {'min': 0, 'max': 60},  # 99%的单次能耗为60（单位：kWh）
    'Trip Duration': {'min': 0, 'max': 580},  # 99%的单上界
    'Trip Distance': {'min': 0, 'max': 220},  # 行程距离（单位：公里）
    'Avg. Speed': {'min': 0, 'max': 120},  # 行程平均速度（单位：km/h）
    'Month': {'min': 1, 'max': 12},  # 当前月份（1到12）
    'Hour': {'min': 0, 'max': 23},  # 当前小时（0到23）
    'Day of the Week': {'min': 0, 'max': 6},  # 当前星期几（1到7）
    'Season': {'min': 1, 'max': 4},  # 当前季节（1：春，2：夏，3：秋，4：冬）
    'Time Period': {'min': 1, 'max': 4},  
    'Is Workday': {'min': 0, 'max': 1},  # 是否工作日（0：否，1：是）

    # Battery Features
    'State of Charge': {'min': 0, 'max': 100},  # 电池的当前状态（0% 到 100%）
    'Accumulated Driving Range': {'min': 0, 'max': 0.4},  # 累积行驶里程（单位：公里）
    'Voltage Range': {'min': 0, 'max': 10},  # 电池电压极差（单位：V）
    'Temperature Range': {'min': 0, 'max': 8},  # 电池温度极差（单位：°C）
    'Insulation': {'min': 0, 'max': 50000},  # 电池绝缘电阻值（单位：MΩ）
    'Historic Avg. EC': {'min': 0, 'max': 30},  # 累计平均能耗（单位：kWh/100km）95%上界，主要是bus

    # Driving Features
    'Energy Recovery Ratio': {'min': 0, 'max': 60},  # 能量回收比例（单位：%）
    'Avg. of Acceleration Pedal': {'min': 0, 'max': 100},  # 加速踏板均值（单位：%）
    'Max of Acceleration Pedal': {'min': 0, 'max': 100},  # 加速踏板最大值（单位：%）
    'Std. of Acceleration Pedal': {'min': 0, 'max': 30},  # 加速踏板标准差（单位：%）
    'Avg. of Brake Pedal': {'min': 0, 'max': 100},  # 制动踏板均值（单位：%）
    'Max of Brake Pedal': {'min': 0, 'max': 100},  # 制动踏板最大值（单位：%）
    'Std. of Brake Pedal': {'min': 0, 'max': 10},  # 制动踏板标准差（单位：%）
    'Avg. of Instantaneous Speed': {'min': 0, 'max': 180},  # 瞬时速度均值（单位：km/h）
    'Max of Instantaneous Speed': {'min': 0, 'max': 180},  # 瞬时速度最大值（单位：km/h）
    'Std. of Instantaneous Speed': {'min': 0, 'max': 35},  # 瞬时速度标准差（单位：km/h）
    'Avg. of Acceleration': {'min': 0, 'max': 2},  # 加速度均值（单位：m/s²）
    'Max of Acceleration': {'min': 0, 'max': 7},  # 加速度最大值（单位：m/s²）
    'Std. of Acceleration': {'min': 0, 'max': 1.5},  # 加速度标准差（单位：m/s²）
    'Avg. of Deceleration': {'min': -5, 'max': 0},  # 减速度均值（单位：m/s²）
    'Max of Deceleration': {'min': -7, 'max': 0},  # 减速度最大值（单位：m/s²）
    'Std. of Deceleration': {'min': 0, 'max': 1.5},  # 减速度标准差（单位：m/s²）

    # Charging Features
    'Equivalent Recharge Count': {'min': 0, 'max': 700},  # 累计充电次数
    'Avg. Recharge Duration': {'min': 0, 'max': 480},  # 平均充电时长（单位：分钟）
    'Cumulative Recharge Duration': {'min': 0, 'max': 72000},  # 累计充电时长（单位：分钟）
    'Max Recharge Duration': {'min': 0, 'max': 720},  # 最大充电时长（单位：分钟）95%
    'Min Recharge Duration': {'min': 0, 'max': 40},  # 最小充电时长（单位：分钟）95%
    'Avg. Starting SOC': {'min': 0, 'max': 70},  # 起始SOC均值（单位：%）
    'Avg. Ending SOC': {'min': 50, 'max': 100},  # 截止SOC均值（单位：%）
    'Avg. Recharge SOC': {'min': 0, 'max': 100},  # 充电SOC均值（单位：%）

    # Environmental Features
    'Temperature': {'min': -12, 'max': 42},  # 平均温度（单位：°C）
    'Air Pressure': {'min': 690, 'max': 790},  # 平均气压（单位：mmHg）
    'Relative Humidity': {'min': 0, 'max': 100},  # 平均相对湿度（单位：%）
    'Wind Speed': {'min': 0, 'max': 12},  # 平均风速（单位：m/s）
    'Visibility': {'min': 0, 'max': 30},  # 平均能见度（单位：km）
    'Avg. Precipitation': {'min': 0, 'max': 150},  # 降水量（单位：mm）

    # Vehicle Features
    'Gross Vehicle Weight': {'min': 1000, 'max': 20000},  # 满载质量（单位：kg）
    'Curb Weight': {'min': 1000, 'max': 12000},  # 整备质量（单位：kg）
    'Vehicle Model': {'min': 1, 'max': 4},  # 车型编号
    'Battery Type': {'min': 1, 'max': 2},  # 电池类型编号
    'Powertrain Type': {'min': 1, 'max': 2},  # 动力类型编号
    'Battery Rated Power': {'min': 10, 'max': 80},  # 电池额定能量（单位：kWh）95%
    'Battery Rated Capacity': {'min': 50, 'max': 180},  # 电池额定容量（单位：Ah）95%
    'Max Power': {'min': 80, 'max': 400},  # 最大功率（单位：kW）
    'Max Torque': {'min': 100, 'max': 700},  # 最大扭矩（单位：Nm）
    'Official ECR': {'min': 10, 'max': 20},  # 官方百公里能耗（单位：kWh/100km）
}

#  === training and testing ===

non_hybrid_data = df[df['Powertrain Type'] == 1]

trip_counts_non_hybrid = non_hybrid_data['car_id'].value_counts()


cars_with_many_trips = trip_counts_non_hybrid[trip_counts_non_hybrid > 1000].index

random.seed(2)
test_car_ids = random.sample(list(cars_with_many_trips), 100)

test_set = df[df['car_id'].isin(test_car_ids)]
# train_set = df[~df['car_id'].isin(test_car_ids)]

# === normalization 两者均可===
for col, bounds in max_min_dict.items():
    if col in test_set.columns:  # 确保列在 DataFrame 中
        min_val = bounds['min']
        max_val = bounds['max']     
        # 应用归一化
        test_set[col] = test_set[col].apply(lambda x: 0 if x < min_val else 
                                          1 if x > max_val else 
                                          (x - min_val) / (max_val - min_val))
# =====  拆分车辆 =======
EV_dict = {}

for car_id, group in test_set.groupby('car_id'):
    # 删除每个子 DataFrame 中的 'car_id' 列
    group = group.drop(columns=['car_id'])
    EV_dict[car_id] = group


#%%
# === 位置编码类 ===
# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 在批次维度上增加维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        device = x.device  # 获取输入的设备
        return self.pe[:, :x.size(1), :].to(device)  # 确保移动到设备

# 属性编码类
class AttributionEmbedding(nn.Module):
    def __init__(self, d_model, Attribution_ids):
        super(AttributionEmbedding, self).__init__()
        self.register_buffer('Attribution_ids', Attribution_ids)  # 使用 register_buffer 保存
        self.d_model = d_model
        
    def forward(self, batch_size, seq_len, device):
        expanded_ids = self.Attribution_ids.unsqueeze(0).expand(batch_size, -1).to(device)
        Attribution_embedding = expanded_ids.unsqueeze(2).expand(batch_size, seq_len, self.d_model)
        return Attribution_embedding


# === Transformer Encoder ===
class TransformerEncoder(nn.Module):
    def __init__(self, Attribution_ids, d_model,  max_len, num_layers,
                nhead, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.feat_embedding = nn.Linear(1, d_model, bias=False)
        self.pos_embedding = PositionalEncoding(d_model, max_len=max_len)
       
        self.Attribution_embedding = AttributionEmbedding(d_model,Attribution_ids)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        batch_first=True,
                                                        dropout=dropout,
                                                        activation="gelu")
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(d_model, d_model)  # 均值映射
        self.fc_logvar = nn.Linear(d_model, d_model)  # 对数方差映射
        self.d_model = d_model
        self.Attribution_ids = Attribution_ids

    def forward(self, src):
        """
        src: (B, F) 输入特征
        """
        B, Feat = src.size()
        embedding_feat = self.feat_embedding(src.unsqueeze(2))  # (B, Feat, 1) -> (B, Feat, d_model)
        #能耗预测任务中并不关键，因此降低权重以加快收敛
        embedding_pos = 0.01* self.pos_embedding(embedding_feat)  # (B, Feat, d_model)     
        embedding_att = 0.01* self.Attribution_embedding(B, Feat, embedding_feat.device)  # 确保设备一致  # (B, Feat, d_model)
        embed_encoder_input = embedding_feat + embedding_pos + embedding_att  # (B, Feat, d_model)
        encoded = self.transformer_encoder(embed_encoder_input)  # (B, Feat, d_model)

        # 隐变量分布参数
        mu = self.fc_mu(encoded.mean(dim=1))  # (B, d_model)
        logvar = self.fc_logvar(encoded.mean(dim=1))  # (B, d_model)
        return encoded, mu, logvar



# === 能耗预测解码头 ===
class EnergyPredictionHead(nn.Module):
    def __init__(self, d_model, feat_dim, dropout=0.2):
        super(EnergyPredictionHead, self).__init__()
        self.bn = nn.BatchNorm1d(feat_dim)  # 批标准化
        self.activation = nn.ReLU()  # 激活函数
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.out_fc1 = nn.Linear(d_model * feat_dim, 1, bias=True)  # 输出层
        self.out_fc2 = nn.Linear(100, feat_dim, bias=True)  # 输出层
        self.out_fc3 = nn.Linear(feat_dim, 1, bias=True)  # 输出层

    def forward(self, encoder_output, **kwargs):
        """
        encoder_output: (B, F, D)
        """
        B, _, _ = encoder_output.size() 

        decoded = encoder_output 

        decoded = self.bn(decoded)      
        decoded = self.activation(decoded)
        decoded= self.dropout(decoded)
        decoded = self.out_fc1(decoded.reshape(B,-1))

        # decoded = self.activation(decoded)
        # decoded= self.dropout(decoded)
        # decoded = self.out_fc2(decoded)
        # decoded = self.activation(decoded)
        # decoded= self.dropout(decoded)
        # decoded = self.out_fc3(decoded)

        # mean_featdim = scaled_.mean(2)

        # decoded = self.out_fc1(decoded.reshape(B,-1))

        # decoded = self.bn(encoder_output)
        # decoded = self.out_fc1(decoded.reshape(B,-1))
        # decoded = self.activation(decoded)
        # decoded = self.bn(encoder_output)
        # decoded = F.softplus(encoder_output)
        # decoded  = self.activation(decoded)
        # decoded= self.dropout(decoded)

        # x = self.out_fc1(encoder_output.reshape(B,-1)) #(batch feat feat_dim)--(batch ,feat_dim)

        # x = self.bn(x)
        # x = self.activation(x)
        # x = self.dropout(x)
        # x = self.out_fc2(decoded.reshape(B,-1))

        return decoded


# 通用模型架构
class GeneralPredictionModel(nn.Module):
    def __init__(self, encoder, decoder):
        """
        encoder: 共享的编码器 (e.g., TransformerEncoder)
        decoder: 解码头模块 (e.g., EnergyPredictionHead, 或其他自定义解码头)
        """
        super(GeneralPredictionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder  # 动态选择解码头

    def forward(self, src, **kwargs):
        """
        src: (B, F) 输入特征
        kwargs: 可选的额外参数传递给解码头（如 `known_features` 等）
        """
        # 编码器生成共享表示
        encoder_output, _, _ = self.encoder(src)  #  Encoder 返回三个输出
        # 解码头生成预测结果
        prediction = self.decoder(encoder_output, **kwargs)
        return prediction


#%%
def fine_tune_model(model, dataloader, criterion, optimizer, scheduler, device, num_epochs=5):
    model.to(device).train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred_y = model(x)
            loss = criterion(pred_y, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

def compute_forward_transfer(EC_True, EC_Pred, dist, chunk_size=16):
    """
    计算 Forward Transfer (FWT)
    """
    ECR_ERR = np.abs(EC_True.reshape(-1, 1) - EC_Pred.reshape(-1, 1)) / dist.reshape(-1, 1)
    chunk_means = [np.mean(ECR_ERR[i:i + chunk_size]) for i in range(0, len(ECR_ERR), chunk_size)]
    return np.mean(np.diff(chunk_means))

def compute_backward_transfer(mae_history, scale_factor, min_val):
    """
    计算 Backward Transfer (BWT)
    """
    BWT_norm = np.mean(np.array(mae_history)[1:] - mae_history[0])
    return BWT_norm * scale_factor + min_val

#%%
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for car_id, ev_data in EV_dict.items():
    # === 构造数据 ===
    
    split_index = int(0.3 * len(ev_data))
    trainX = torch.Tensor(ev_data.iloc[:split_index, 1:].values).float()
    trainY = torch.Tensor(ev_data.iloc[:split_index, 0].values).float()
    testX = torch.Tensor(ev_data.iloc[split_index:, 1:].values).float()
    testY = torch.Tensor(ev_data.iloc[split_index:, 0].values).float()
    dist = ev_data.iloc[split_index:, 2].values

    train_dataset = Data.TensorDataset(trainX, trainY)
    test_dataset = Data.TensorDataset(testX, testY)

    Dataloaders_train = Data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    Dataloaders_test = Data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型和优化器初始化
    #  === 超参数设置 ===

    d_model = 256
    feat_dim = trainX.shape[1]
    max_len = 100
    num_layers = 6
    nhead = 4
    dim_feedforward = 1024
    dropout = 0.1

    # === 模型初始化 ===
    # 初始化多个 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Attribution_ids = torch.tensor([5] * 9 + [4] * 6 + [3] * 16 + [2] * 8 + [1] * 6 + [0] * 10)
    min_val = Attribution_ids.min()
    max_val = Attribution_ids.max()
    normalized_attribution_ids = (Attribution_ids - min_val) / (max_val - min_val)


    #  ====== 加载参数 ======

    encoder_pretrained = TransformerEncoder(normalized_attribution_ids.to(device), d_model=d_model, 
                                            max_len=max_len, num_layers=num_layers, nhead=nhead, 
                                            dim_feedforward=dim_feedforward, dropout=dropout)
 
    # 解码头
    energy_head = EnergyPredictionHead(d_model=d_model, feat_dim=feat_dim, 
                                       dropout=dropout).to(device)

    # ===== 创建能耗预测模型 =======
    ECM = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)

    ECM.load_state_dict(new_state_dict)
    ECM.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(ECM.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    # 微调阶段
    fine_tune_model(ECM, Dataloaders_train, criterion, optimizer, scheduler, device)

    #% 在线学习阶段
    online_optimizer = torch.optim.AdamW(ECM.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4)
    online_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(online_optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    preds, reals, old_task_mae = [], [], []
    start_time = time.time()

    #在历史数据和当前任务共同参与下学习
    combined_x,combined_y=[trainX.to(device)],[trainY.unsqueeze(1).to(device)]
    for step, (x, y) in enumerate(Dataloaders_test):
        x, y = x.to(device), y.to(device)
        ECM.eval()
        with torch.no_grad():
            pred_y = ECM(x)
            preds.append(pred_y.cpu().detach())
            reals.append(y.cpu().detach())
            old_task_mae.append(evaluation(ECM(trainX.to(device)).detach().cpu().numpy(),
                                           trainY.numpy())[0])

        #这是每一个chunk的数据
        combined_x.append(x)
        combined_y.append(y.unsqueeze(1))

        # 然后在第一维上拼接张量
        cx = torch.cat(combined_x, dim=0)
        cy = torch.cat(combined_y, dim=0)
        
        comb_dataset = Data.TensorDataset(cx,cy)
        Dataloaders_comb = Data.DataLoader(comb_dataset, batch_size=16, shuffle=False,
                                            generator=torch.Generator().manual_seed(42))

        chunk_loss = 0.0
        num_epochs = 1
        for epoch in range(num_epochs):
            ECM.train()
            total_loss=0.0
            for i, (n_x, n_y) in enumerate(Dataloaders_comb):

                n_x, n_y = n_x.to(device), n_y.to(device)

                online_optimizer.zero_grad()

                pred_y = ECM(n_x)
                loss = criterion(pred_y, n_y)
                loss.backward()
                online_optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(Dataloaders_comb)
            chunk_loss += avg_loss

        online_scheduler.step(chunk_loss / num_epochs)
        print(f"Chunk: {step + 1}/{len(Dataloaders_test)}, chunk_loss: {chunk_loss:.4f}")

    end_time = time.time()
    execution_time = (end_time - start_time) / len(testX)

    targets = torch.cat(reals, dim=0).numpy()
    predictions = torch.cat(preds, dim=0).numpy()

    EC_True = targets * 60
    EC_Pred = np.abs(predictions) * 60

    DirectMeasure = np.array(evaluation(EC_True[1:], EC_Pred[1:]))
    FWT = compute_forward_transfer(EC_True, EC_Pred, dist)
    BWT = compute_backward_transfer(old_task_mae,60, 0)

    print(f"Direct Measure: {DirectMeasure}, FWT: {FWT}, BWT: {BWT}, Execution Time: {execution_time:.4f} seconds")
    
    
    # 运行一次
    break

