#%% -*- coding: utf-8 -*-
from attr import s
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.utils.data as Data
import math
import os
import random
import torch.nn.functional as F
#%% load
all_df = pd.read_pickle('/home/ps/haichao/1-lifelong_learning/trip data/all_df.pkl')
"""Encoder for categorical variables"""
def labeling(df):
    
    # Label Encoding for travel season
    season_mapping = {'spring': 1, 'summer': 2, 'autumn': 3, 'winter': 4}
    df['当前季节'] = df['当前季节'].map(season_mapping).astype(int)
    # Label Encoding for travel period
    period_mapping = {'morning peak': 1, 'night peak': 2, 'other time': 3, "night time": 4}
    df['当前时段'] = df['当前时段'].map(period_mapping).astype(int)
    week_mapping = {'weekday': 0, 'weekend': 1}
    df['当前是否工作日'] = df['当前是否工作日'].map(week_mapping).astype(int)
    # Label Encoding for vehicle type
    vehicle_mapping = {'Sedan': 1, 'SUV': 2,'客车-公交': 3,'物流': 4}
    df['车型'] = df['车型'].map(vehicle_mapping).fillna(5).astype(int)
    # Label Encoding for battery type
    battery_mapping = {'三元材料电池': 1, '磷酸铁锂电池': 2}
    df['电池类型'] = df['电池类型'].map(battery_mapping).fillna(3).astype(int)
    # Label Encoding for power type
    power_mapping = {'BEV': 1, 'PHEV': 2}
    df['动力类型'] = df['动力类型'].map(power_mapping).fillna(3).astype(int)
    return df
labeled_df=labeling(all_df)
"Replace with English names"
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
df = df.drop('car_id', axis=1)
# Fill NaN with median
df = df.apply(lambda x: x.fillna(x.median()), axis=0)
# nan_count = df.isna().sum().sum()
# # Calculate total data count
# total_count = df.size
# # Calculate the ratio of NaN to total data
# nan_ratio = nan_count / total_count
# print(f"NaN ratio: {nan_ratio:.2%}")
#%%
max_min_dict = {
    # Trip Features  Mostly take the 99% upper bound
    'Energy Consumption': {'min': 0, 'max': 60},  # 99% of single trip energy consumption is 60 (unit: kWh)
    'Trip Duration': {'min': 0, 'max': 580},  # 99% upper bound
    'Trip Distance': {'min': 0, 'max': 220},  # Trip distance (unit: km)
    'Avg. Speed': {'min': 0, 'max': 120},  # Average speed (unit: km/h)
    'Month': {'min': 1, 'max': 12},  # Current month (1 to 12)
    'Hour': {'min': 0, 'max': 23},  # Current hour (0 to 23)
    'Day of the Week': {'min': 0, 'max': 6},  # Current day of the week (1 to 7)
    'Season': {'min': 1, 'max': 4},  # Current season (1: spring, 2: summer, 3: autumn, 4: winter)
    'Time Period': {'min': 1, 'max': 4},  
    'Is Workday': {'min': 0, 'max': 1},  # Whether it's a workday (0: no, 1: yes)
    # Battery Features
    'State of Charge': {'min': 0, 'max': 100},  # Current state of battery (0% to 100%)
    'Accumulated Driving Range': {'min': 0, 'max': 0.4},  # Accumulated driving range (unit: km)
    'Voltage Range': {'min': 0, 'max': 10},  # Battery voltage range (unit: V)
    'Temperature Range': {'min': 0, 'max': 8},  # Battery temperature range (unit: °C)
    'Insulation': {'min': 0, 'max': 50000},  # Battery insulation resistance (unit: MΩ)
    'Historic Avg. EC': {'min': 0, 'max': 30},  # Cumulative average energy consumption (unit: kWh/100km) 95% upper bound, mainly for bus
    # Driving Features
    'Energy Recovery Ratio': {'min': 0, 'max': 60},  # Energy recovery ratio (unit: %)
    'Avg. of Acceleration Pedal': {'min': 0, 'max': 100},  # Average acceleration pedal (unit: %)
    'Max of Acceleration Pedal': {'min': 0, 'max': 100},  # Maximum acceleration pedal (unit: %)
    'Std. of Acceleration Pedal': {'min': 0, 'max': 30},  # Standard deviation of acceleration pedal (unit: %)
    'Avg. of Brake Pedal': {'min': 0, 'max': 100},  # Average brake pedal (unit: %)
    'Max of Brake Pedal': {'min': 0, 'max': 100},  # Maximum brake pedal (unit: %)
    'Std. of Brake Pedal': {'min': 0, 'max': 10},  # Standard deviation of brake pedal (unit: %)
    'Avg. of Instantaneous Speed': {'min': 0, 'max': 180},  # Average instantaneous speed (unit: km/h)
    'Max of Instantaneous Speed': {'min': 0, 'max': 180},  # Maximum instantaneous speed (unit: km/h)
    'Std. of Instantaneous Speed': {'min': 0, 'max': 35},  # Standard deviation of instantaneous speed (unit: km/h)
    'Avg. of Acceleration': {'min': 0, 'max': 2},  # Average acceleration (unit: m/s²)
    'Max of Acceleration': {'min': 0, 'max': 7},  # Maximum acceleration (unit: m/s²)
    'Std. of Acceleration': {'min': 0, 'max': 1.5},  # Standard deviation of acceleration (unit: m/s²)
    'Avg. of Deceleration': {'min': -5, 'max': 0},  # Average deceleration (unit: m/s²)
    'Max of Deceleration': {'min': -7, 'max': 0},  # Maximum deceleration (unit: m/s²)
    'Std. of Deceleration': {'min': 0, 'max': 1.5},  # Standard deviation of deceleration (unit: m/s²)
    # Charging Features
    'Equivalent Recharge Count': {'min': 0, 'max': 700},  # Cumulative recharge count
    'Avg. Recharge Duration': {'min': 0, 'max': 480},  # Average recharge duration (unit: minutes)
    'Cumulative Recharge Duration': {'min': 0, 'max': 72000},  # Cumulative recharge duration (unit: minutes)
    'Max Recharge Duration': {'min': 0, 'max': 720},  # Maximum recharge duration (unit: minutes) 95%
    'Min Recharge Duration': {'min': 0, 'max': 40},  # Minimum recharge duration (unit: minutes) 95%
    'Avg. Starting SOC': {'min': 0, 'max': 70},  # Average starting SOC (unit: %)
    'Avg. Ending SOC': {'min': 50, 'max': 100},  # Average ending SOC (unit: %)
    'Avg. Recharge SOC': {'min': 0, 'max': 100},  # Average recharge SOC (unit: %)
    # Environmental Features
    'Temperature': {'min': -12, 'max': 42},  # Average temperature (unit: °C)
    'Air Pressure': {'min': 690, 'max': 790},  # Average air pressure (unit: mmHg)
    'Relative Humidity': {'min': 0, 'max': 100},  # Average relative humidity (unit: %)
    'Wind Speed': {'min': 0, 'max': 12},  # Average wind speed (unit: m/s)
    'Visibility': {'min': 0, 'max': 30},  # Average visibility (unit: km)
    'Avg. Precipitation': {'min': 0, 'max': 150},  # Precipitation (unit: mm)
    # Vehicle Features
    'Gross Vehicle Weight': {'min': 1000, 'max': 20000},  # Gross vehicle weight (unit: kg)
    'Curb Weight': {'min': 1000, 'max': 12000},  # Curb weight (unit: kg)
    'Vehicle Model': {'min': 1, 'max': 4},  # Vehicle model number
    'Battery Type': {'min': 1, 'max': 2},  # Battery type number
    'Powertrain Type': {'min': 1, 'max': 2},  # Powertrain type number
    'Battery Rated Power': {'min': 10, 'max': 80},  # Battery rated power (unit: kWh) 95%
    'Battery Rated Capacity': {'min': 50, 'max': 180},  # Battery rated capacity (unit: Ah) 95%
    'Max Power': {'min': 80, 'max': 400},  # Maximum power (unit: kW)
    'Max Torque': {'min': 100, 'max': 700},  # Maximum torque (unit: Nm)
    'Official ECR': {'min': 10, 'max': 20},  # Official energy consumption rate (unit: kWh/100km)
}
# Assuming df is your DataFrame
for col, bounds in max_min_dict.items():
    if col in df.columns:  # Ensure the column is in the DataFrame
        min_val = bounds['min']
        max_val = bounds['max']     
        # Apply normalization
        df[col] = df[col].apply(lambda x: 0 if x < min_val else 
                                          1 if x > max_val else 
                                          (x - min_val) / (max_val - min_val))

#%%
# === Positional Encoding Class ===
# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add dimension on batch axis
        self.register_buffer('pe', pe)
    def forward(self, x):
        device = x.device  # Get the device of the input
        return self.pe[:, :x.size(1), :].to(device)  # Ensure moving to device
# Attribution Encoding Class
class AttributionEmbedding(nn.Module):
    def __init__(self, d_model, Attribution_ids):
        super(AttributionEmbedding, self).__init__()
        self.register_buffer('Attribution_ids', Attribution_ids)  # Save using register_buffer
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
        self.fc_mu = nn.Linear(d_model, d_model)  # Mean mapping
        self.fc_logvar = nn.Linear(d_model, d_model)  # Log variance mapping
        self.d_model = d_model
        self.Attribution_ids = Attribution_ids
    def forward(self, src):
        """
        src: (B, F) Input features
        """
        B, Feat = src.size()
        embedding_feat = self.feat_embedding(src.unsqueeze(2))  # (B, Feat, 1) -> (B, Feat, d_model)
        embedding_pos = self.pos_embedding(embedding_feat)  # (B, Feat, d_model)     
        embedding_att = self.Attribution_embedding(B, Feat, embedding_feat.device)  # Ensure device consistency  # (B, Feat, d_model)
        embed_encoder_input = embedding_feat + embedding_pos + embedding_att  # (B, Feat, d_model)
        encoded = self.transformer_encoder(embed_encoder_input)  # (B, Feat, d_model)
        # Latent variable distribution parameters
        mu = self.fc_mu(encoded.mean(dim=1))  # (B, d_model)
        logvar = self.fc_logvar(encoded.mean(dim=1))  # (B, d_model)
        return encoded, mu, logvar

# === Transformer Decoder ===
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, max_len, num_layers, nhead, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()
        self.feat_embedding = nn.Linear(1, d_model, bias=False)
        self.pos_embedding = PositionalEncoding(d_model, max_len=max_len)
        self.Attribution_embedding = AttributionEmbedding(d_model,Attribution_ids)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        batch_first=True,
                                                        dropout=dropout,
                                                        activation="gelu")
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)  # Output single feature value
        self.bn = nn.BatchNorm1d(55)
        self.dropout = nn.Dropout(0.1)  # Add Dropout layer
    def forward(self, z, src):
        """
        z: (B, d_model) Latent variable
        src: (B, F) Input features, used as condition
        """
        B, Feat= src.size()
    
        z_expanded = z.unsqueeze(1).expand(-1, Feat, -1)  # (B, F, d_model)
        embedding_feat = self.feat_embedding(src.unsqueeze(2))  # (B, F, 1) -> (B, F, d_model)
        embedding_pos = self.pos_embedding(embedding_feat)  # (B, F, d_model)     
        embedding_att = self.Attribution_embedding(B, Feat, embedding_feat.device)  # (B, F, d_model)
        decoder_input = embedding_feat + embedding_pos + embedding_att  # (B, F, d_model)
        decoded = self.transformer_decoder(tgt=z_expanded, memory=decoder_input)  # (B, F, d_model)
        
        
        decoded = self.bn(decoded)
        decoded = F.softplus(decoded)
        decoded= self.dropout(decoded)
        output = self.output_layer(decoded).squeeze(-1)  # (B, F)
        
        return output

# === VAE Model ===
class TransformerVAE(nn.Module):
    def __init__(self, encoder, decoder, d_model=512):
        super(TransformerVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model
    def reparameterize(self, mu, logvar):
        """Reparameterize to sample latent variable z"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, src):
        """
        src: (B, F) Input features
        """
        # Encoding phase
        _, mu, logvar = self.encoder(src)
        z = self.reparameterize(mu, logvar)
        # Decoding phase
        reconstructed = self.decoder(z, src)
        
        return reconstructed, mu, logvar
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')

#%%#
# === Data ===
# Shuffle the rows
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
trainX = torch.Tensor(shuffled_df.iloc[:,1:].values).float()
train_dataset = Data.TensorDataset(trainX)
Dataloaders_train = Data.DataLoader(dataset=train_dataset,
                                    batch_size=256, shuffle=True,
                                    generator=torch.Generator().manual_seed(42))
# 
#  === Hyperparameter Settings ===
d_model = 256
feat_dim = trainX.shape[1]
max_len = 100
num_layers = 6
nhead = 4
dim_feedforward = 1024
dropout = 0.1

# === Model Initialization ===
# Initialize multiple GPUs
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Specify visible GPUs
Attribution_ids = torch.tensor([5] * 9 + [4] * 6 + [3] * 16 + [2] * 8 + [1] * 6 + [0] * 10)
min_val = Attribution_ids.min()
max_val = Attribution_ids.max()
normalized_attribution_ids = (Attribution_ids - min_val) / (max_val - min_val)

encoder = TransformerEncoder(normalized_attribution_ids.to(device), d_model=d_model, max_len=max_len, 
                             num_layers=num_layers, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
decoder = TransformerDecoder(d_model=d_model,  max_len=max_len, num_layers=num_layers, nhead=nhead,
                              dim_feedforward=dim_feedforward, dropout=dropout).to(device)
EVVAE = TransformerVAE(encoder, decoder, d_model=d_model)
EVVAE = EVVAE.to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    EVVAE = nn.DataParallel(EVVAE)  # Use DataParallel to wrap the model
else:
    print("Let's use single GPU!")
EVVAE.apply(initialize_weights)
# === Loss Function and Optimizer ===
def vae_loss(reconstructed, original, mu, logvar , beta=0.05):
    recon_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_divergence /original.size(0)

optimizer = torch.optim.AdamW(EVVAE.parameters(), lr=1e-3,
                              betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                 T_0=10, T_mult=2, 
                                                                 eta_min=1e-6)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                        patience=10, factor=0.99)
# Set initial epoch and total_loss

# Checkpoint path and save interval
models_folder = '/home/ps/haichao/1-lifelong_learning/Model'
# os.makedirs(models_folder, exist_ok=True)
# checkpoint_path = os.path.join(models_folder, f'model_checkpoint_epoch{epoch+1}.pt')
# save_interval = 1000  # Save the model every 1000 steps
# # Check if a previous checkpoint exists
# if os.path.exists(checkpoint_path):
#     if torch.cuda.is_available():
#         checkpoint = torch.load(checkpoint_path)
#     else:
#         checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#     if 'epoch' in checkpoint and 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
#         start_epoch = checkpoint['epoch'] + 1
#         EVVAE.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         total_loss = checkpoint['total_loss']
        # Feat_Representation=checkpoint['Feat_Representation'] 

#%%
best_loss = float('inf')  # Initialize minimum validation loss to infinity
start_epoch = 0
# === Training Process ===
for epoch in range(start_epoch, 10000):
    
    EVVAE.train()
    total_loss = 0.0
    total_loss_epoch = 0.0
    batch = 0
    log_interval = max(1, len(Dataloaders_train) // 5)  # Prevent log_interval from being 0
    for batch, (Original_data,) in enumerate(Dataloaders_train):
        Original_data = Original_data.to(device)
        optimizer.zero_grad()
        reconstructed, mu, logvar = EVVAE(Original_data)
        loss = vae_loss(reconstructed, Original_data, mu, logvar)
        loss.backward()
        gradients = []
        for param in EVVAE.parameters():
            if param.grad is not None:
                gradients.append(param.grad.norm().item())
        torch.nn.utils.clip_grad_norm_(EVVAE.parameters(), max_norm=5)
        optimizer.step()
        total_loss += loss.item()
        total_loss_epoch += loss.item()
        if (batch + 1) % log_interval == 0 and (batch + 1) > 0:
            cur_loss = total_loss / len(reconstructed) / 5
            mean_gradient = torch.mean(torch.tensor(gradients))
            print('| epoch {:3d} | {:5d}/{:5d} batches | ''lr {:02.9f} | ''loss {:5.5f} | ''mean gradient {:5.5f}'
                  .format(epoch, (batch + 1), len(Dataloaders_train) ,
                          optimizer.param_groups[-1]['lr'], cur_loss, mean_gradient))
            
            total_loss = 0
    # scheduler.step( total_loss_epoch/len(Dataloaders_train.dataset) )
    scheduler.step()
    # If the current validation loss is smaller, save the best model
    if (total_loss_epoch/len(Dataloaders_train.dataset))  < best_loss:
        best_loss = (total_loss_epoch/len(Dataloaders_train.dataset)) 
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': EVVAE.state_dict(),
            "encoder_state_dict" : EVVAE.module.encoder.state_dict(),  # Get encoder weights
            "ecoder_state_dict" : EVVAE.module.decoder.state_dict(),  # Get decoder weights
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        torch.save(checkpoint, os.path.join(models_folder, 'EVVAE.pt'))
        print(f"Best model saved at epoch {epoch + 1} with loss {best_loss:.6f}")
    torch.cuda.empty_cache()
    if (epoch + 1) % 5 == 0:
        print('-' * 89)
        print(f"Epoch {epoch + 1}: Learning Rate: {optimizer.param_groups[-1]['lr']:.6f}, Loss: {total_loss_epoch / len(Dataloaders_train.dataset):.9f}")
        print('-' * 89)
        print(f"Reconstructed: {reconstructed[-1]}")
        print(f"Original: {Original_data[-1]}")
    # Save checkpoint
    if (epoch + 1) % 1000 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': EVVAE.state_dict(),
            "encoder_state_dict" : EVVAE.module.encoder.state_dict(),  # Get encoder weights
            "ecoder_state_dict" : EVVAE.module.decoder.state_dict(),  # Get decoder weights
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(models_folder, 
                                            f'EVVAE_checkpoint_epoch_{epoch+1}.pt'))