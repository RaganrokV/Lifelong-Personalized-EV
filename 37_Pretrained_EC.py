#%%
import random
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.utils.data as Data
import math
import torch.nn.parallel
import os
#%%
all_df = pd.read_pickle('/home/ps/haichao/1-lifelong_learning/trip data/all_df.pkl')
#%%
"""Encode categorical variables"""
def labeling(df):
    
    # Label Encoding for season
    season_mapping = {'spring': 1, 'summer': 2, 'autumn': 3, 'winter': 4}
    df['当前季节'] = df['当前季节'].map(season_mapping).astype(int)
    # Label Encoding for time period
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
    # Label Encoding for powertrain type
    power_mapping = {'BEV': 1, 'PHEV': 2}
    df['动力类型'] = df['动力类型'].map(power_mapping).fillna(3).astype(int)
    return df
labeled_df=labeling(all_df)
"Rename columns to English"
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
    lambda x: x.fillna(x.median()), axis=0)  # Fill missing values with median or 0

#%%
max_min_dict = {
    # Trip Features  Mostly using 99% upper bound
    'Energy Consumption': {'min': 0, 'max': 60},  # 99% of single trip energy consumption is 60 (unit: kWh)
    'Trip Duration': {'min': 0, 'max': 580},  # 99% upper bound
    'Trip Distance': {'min': 0, 'max': 220},  # Trip distance (unit: km)
    'Avg. Speed': {'min': 0, 'max': 120},  # Average trip speed (unit: km/h)
    'Month': {'min': 1, 'max': 12},  # Current month (1 to 12)
    'Hour': {'min': 0, 'max': 23},  # Current hour (0 to 23)
    'Day of the Week': {'min': 0, 'max': 6},  # Current day of the week (1 to 7)
    'Season': {'min': 1, 'max': 4},  # Current season (1: Spring, 2: Summer, 3: Autumn, 4: Winter)
    'Time Period': {'min': 1, 'max': 4},  
    'Is Workday': {'min': 0, 'max': 1},  # Is it a workday? (0: No, 1: Yes)
    # Battery Features
    'State of Charge': {'min': 0, 'max': 100},  # Current battery state (0% to 100%)
    'Accumulated Driving Range': {'min': 0, 'max': 0.4},  # Accumulated driving range (unit: km)
    'Voltage Range': {'min': 0, 'max': 10},  # Battery voltage range (unit: V)
    'Temperature Range': {'min': 0, 'max': 8},  # Battery temperature range (unit: °C)
    'Insulation': {'min': 0, 'max': 50000},  # Battery insulation resistance (unit: MΩ)
    'Historic Avg. EC': {'min': 0, 'max': 30},  # Historic average energy consumption (unit: kWh/100km) 95% upper bound, mainly for buses
    # Driving Features
    'Energy Recovery Ratio': {'min': 0, 'max': 60},  # Energy recovery ratio (unit: %)
    'Avg. of Acceleration Pedal': {'min': 0, 'max': 100},  # Average of acceleration pedal (unit: %)
    'Max of Acceleration Pedal': {'min': 0, 'max': 100},  # Maximum of acceleration pedal (unit: %)
    'Std. of Acceleration Pedal': {'min': 0, 'max': 30},  # Standard deviation of acceleration pedal (unit: %)
    'Avg. of Brake Pedal': {'min': 0, 'max': 100},  # Average of brake pedal (unit: %)
    'Max of Brake Pedal': {'min': 0, 'max': 100},  # Maximum of brake pedal (unit: %)
    'Std. of Brake Pedal': {'min': 0, 'max': 10},  # Standard deviation of brake pedal (unit: %)
    'Avg. of Instantaneous Speed': {'min': 0, 'max': 180},  # Average of instantaneous speed (unit: km/h)
    'Max of Instantaneous Speed': {'min': 0, 'max': 180},  # Maximum of instantaneous speed (unit: km/h)
    'Std. of Instantaneous Speed': {'min': 0, 'max': 35},  # Standard deviation of instantaneous speed (unit: km/h)
    'Avg. of Acceleration': {'min': 0, 'max': 2},  # Average of acceleration (unit: m/s²)
    'Max of Acceleration': {'min': 0, 'max': 7},  # Maximum of acceleration (unit: m/s²)
    'Std. of Acceleration': {'min': 0, 'max': 1.5},  # Standard deviation of acceleration (unit: m/s²)
    'Avg. of Deceleration': {'min': -5, 'max': 0},  # Average of deceleration (unit: m/s²)
    'Max of Deceleration': {'min': -7, 'max': 0},  # Maximum of deceleration (unit: m/s²)
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
    'Vehicle Model': {'min': 1, 'max': 4},  # Vehicle model code
    'Battery Type': {'min': 1, 'max': 2},  # Battery type code
    'Powertrain Type': {'min': 1, 'max': 2},  # Powertrain type code
    'Battery Rated Power': {'min': 10, 'max': 80},  # Battery rated power (unit: kWh) 95%
    'Battery Rated Capacity': {'min': 50, 'max': 180},  # Battery rated capacity (unit: Ah) 95%
    'Max Power': {'min': 80, 'max': 400},  # Maximum power (unit: kW)
    'Max Torque': {'min': 100, 'max': 700},  # Maximum torque (unit: Nm)
    'Official ECR': {'min': 10, 'max': 20},  # Official energy consumption rate (unit: kWh/100km)
}
#%%
# === Positional Encoding Class ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    def forward(self, x):
        device = x.device  # Get the device of the input
        return self.pe[:, :x.size(1), :].to(device)  # Ensure it's moved to the device
# Attribution Embedding Class
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
        # Not critical in energy consumption prediction, so reduce weight to speed up convergence
        embedding_pos = 0.01* self.pos_embedding(embedding_feat)  # (B, Feat, d_model)     
        embedding_att = 0.01* self.Attribution_embedding(B, Feat, embedding_feat.device)  # Ensure device consistency  # (B, Feat, d_model)
        embed_encoder_input = embedding_feat + embedding_pos + embedding_att  # (B, Feat, d_model)
        encoded = self.transformer_encoder(embed_encoder_input)  # (B, Feat, d_model)
        # Latent variable distribution parameters
        mu = self.fc_mu(encoded.mean(dim=1))  # (B, d_model)
        logvar = self.fc_logvar(encoded.mean(dim=1))  # (B, d_model)
        return encoded, mu, logvar

# === Energy Prediction Head ===
class EnergyPredictionHead(nn.Module):
    def __init__(self, d_model, feat_dim, dropout=0.2):
        super(EnergyPredictionHead, self).__init__()
        self.bn = nn.BatchNorm1d(feat_dim)  # Batch normalization
        self.activation = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.out_fc1 = nn.Linear(d_model * feat_dim, 1, bias=True)  # Output layer
        self.out_fc2 = nn.Linear(100, feat_dim, bias=True)  # Output layer
        self.out_fc3 = nn.Linear(feat_dim, 1, bias=True)  # Output layer
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
        return decoded

# General Prediction Model
class GeneralPredictionModel(nn.Module):
    def __init__(self, encoder, decoder):
        """
        encoder: Shared encoder (e.g., TransformerEncoder)
        decoder: Decoder head module (e.g., EnergyPredictionHead, or other custom decoder)
        """
        super(GeneralPredictionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder  # Dynamically select decoder
    def forward(self, src, **kwargs):
        """
        src: (B, F) Input features
        kwargs: Optional additional parameters passed to the decoder (e.g., `known_features`)
        """
        # Encoder generates shared representation
        encoder_output, _, _ = self.encoder(src)  # Encoder returns three outputs
        # Decoder generates prediction
        prediction = self.decoder(encoder_output, **kwargs)
        return prediction

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight)  # Kaiming initialization

def reparameterization(model, pretrained_model):
    """
    Initialize with pre-trained parameters (mean & std)
    """
    for (name_new, param_new), (name_old, param_old) in zip(model.named_parameters(), pretrained_model.named_parameters()):
        if param_old.requires_grad:
            with torch.no_grad():
                mean, std = param_old.mean(), param_old.std()
                param_new.data = torch.normal(mean, std, size=param_new.shape, device=param_new.device)
                
#%%
#  === Training and Testing ===
non_hybrid_data = df[df['Powertrain Type'] == 1]
trip_counts_non_hybrid = non_hybrid_data['car_id'].value_counts()

cars_with_many_trips = trip_counts_non_hybrid[trip_counts_non_hybrid > 1000].index
random.seed(2)
test_car_ids = random.sample(list(cars_with_many_trips), 100)
test_set = df[df['car_id'].isin(test_car_ids)]
train_set = df[~df['car_id'].isin(test_car_ids)]
#%% 
# === Normalization (Both methods are acceptable) ===
train_set = train_set.drop('car_id', axis=1)
# train_set = train_set[(train_set['Energy Consumption'] / train_set['Trip Distance'])<= 0.2]
# train_set=train_set.iloc[:10000,:]
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# train_set_normalized = scaler.fit_transform(train_set)
# # Convert normalized data back to DataFrame
# train_set_normalized = pd.DataFrame(train_set_normalized, columns=train_set.columns)
# shuffled_df = train_set_normalized.sample(frac=1, random_state=42).reset_index(drop=True)

for col, bounds in max_min_dict.items():
    if col in train_set.columns:  # Ensure the column exists in the DataFrame
        min_val = bounds['min']
        max_val = bounds['max']     
        # Apply normalization
        train_set[col] = train_set[col].apply(lambda x: 0 if x < min_val else 
                                          1 if x > max_val else 
                                          (x - min_val) / (max_val - min_val))
# === Construct Data ===
shuffled_df = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
trainX = torch.Tensor(shuffled_df.iloc[:,1:].values).float()
trainY = torch.Tensor(shuffled_df.iloc[:,0].values).float()
train_dataset = Data.TensorDataset(trainX,trainY)
Dataloaders_train = Data.DataLoader(dataset=train_dataset,
                                    batch_size=256, shuffle=True,
                                    generator=torch.Generator().manual_seed(42))
#%%
#  === Hyperparameters Setup ===
d_model = 256
feat_dim = trainX.shape[1]
max_len = 100
num_layers = 6
nhead = 4
dim_feedforward = 1024
dropout = 0.1
# ==== Can be the same or adjusted ====
d_model_reparameterization= 256
feat_dim_reparameterization = trainX.shape[1]
max_len_reparameterization = 100
num_layers_reparameterization = 6
nhead_reparameterization = 4
dim_feedforward_reparameterization = 1023
dropout_reparameterization = 0.1
# === Model Initialization ===
# Initialize multiple GPUs
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Specify visible GPUs
Attribution_ids = torch.tensor([5] * 9 + [4] * 6 + [3] * 16 + [2] * 8 + [1] * 6 + [0] * 10)
min_val = Attribution_ids.min()
max_val = Attribution_ids.max()
normalized_attribution_ids = (Attribution_ids - min_val) / (max_val - min_val)

#  ====== Load parameters and reparameterize ======
encoder_pretrained = TransformerEncoder(normalized_attribution_ids.to(device), d_model=d_model, 
                                        max_len=max_len, num_layers=num_layers, nhead=nhead, 
                                        dim_feedforward=dim_feedforward, dropout=dropout)
encoder_reparameterization = TransformerEncoder(normalized_attribution_ids.to(device), d_model=d_model_reparameterization, 
                                    max_len=max_len_reparameterization, num_layers=num_layers_reparameterization, nhead=nhead_reparameterization, 
                                    dim_feedforward=dim_feedforward_reparameterization, dropout=dropout_reparameterization)

enc_para = torch.load('/home/ps/haichao/1-lifelong_learning/Model/model_checkpoint_epoch_2000.pt')
encoder_pretrained.load_state_dict(enc_para['encoder_state_dict'])
# encoder_random.apply(initialize_weights)
reparameterization(encoder_reparameterization, encoder_pretrained)  

# Initialize decoder head
energy_head = EnergyPredictionHead(d_model=d_model_reparameterization, feat_dim=feat_dim_reparameterization, 
                                   dropout=dropout_reparameterization).to(device)
energy_head.apply(initialize_weights)
# ===== Create Energy Consumption Model =======
ECM = GeneralPredictionModel(encoder_reparameterization, energy_head).to(device)

ECM.to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    ECM = nn.DataParallel(ECM)  # Wrap the model with DataParallel
else:
    print("Let's use single GPU!")
# optimizer = torch.optim.AdamW(ECM.parameters(), lr=1e-5,
#                               betas=(0.9, 0.999), weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
# patience=10, factor=0.99)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
#                                                                  T_0=10, T_mult=2, 
#                                                                  eta_min=1e-7)
# === Learning Rate Scheduler (Asynchronous Update) ===
encoder_lr = 1e-5  # Encoder learning rate
decoder_lr = 1e-4  # Decoder learning rate
optimizer = torch.optim.AdamW([
    {'params': ECM.module.encoder.parameters(), 'lr': encoder_lr},  # Lower learning rate for encoder
    {'params': ECM.module.decoder.parameters(), 'lr': decoder_lr},  # Higher learning rate for decoder
], betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                 T_0=10, T_mult=2, 
                                                                 eta_min=1e-6)                                                             
criterion = nn.MSELoss()
#%%
models_folder = '/home/ps/haichao/1-lifelong_learning/Model'
best_loss = float('inf')  # Initialize minimum validation loss as infinity
start_epoch = 0
# === Training Process ===
for epoch in range(start_epoch, 10000):
    
    ECM.train()
    total_loss = 0.0
    total_loss_epoch = 0.0
    batch = 0
    log_interval = max(1, len(Dataloaders_train) // 5)  # Prevent log_interval from being 0
    for batch, (x,y) in enumerate(Dataloaders_train):
        x,y= x.to(device),y.to(device)
        optimizer.zero_grad()
        predy = ECM(x)
        loss = criterion(predy, y.unsqueeze(1)) #If dimensions mismatch, no error but results will differ!!!
        loss.backward()
        gradients = []
        for param in ECM.parameters():
            if param.grad is not None:
                gradients.append(param.grad.norm().item())
        torch.nn.utils.clip_grad_norm_(ECM.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()
        total_loss_epoch += loss.item()
        if (batch + 1) % log_interval == 0 and (batch + 1) > 0:
            cur_loss = total_loss / len(x) / 5
            mean_gradient = torch.mean(torch.tensor(gradients))
            print('| epoch {:3d} | {:5d}/{:5d} batches | ''lr {:02.9f} | ''loss {:5.5f} | ''mean gradient {:5.5f}'
                  .format(epoch, (batch + 1), len(Dataloaders_train) ,
                          optimizer.param_groups[-1]['lr'], cur_loss, mean_gradient))
            
            total_loss = 0
    # scheduler.step(total_loss)
    scheduler.step()
    # Save the best model if the current validation loss is smaller
    if (total_loss_epoch/len(Dataloaders_train.dataset))  < best_loss:
        best_loss = (total_loss_epoch/len(Dataloaders_train.dataset)) 
        checkpoint = {
            'epoch': epoch,
            'model_state_dict':  ECM.module.state_dict(),
            "encoder_state_dict" : ECM.module.encoder.state_dict(),  # Get encoder weights
            "ecoder_state_dict" : ECM.module.decoder.state_dict(),  # Get decoder weights
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        # torch.save(checkpoint, os.path.join(models_folder, 'best_ECM.pt'))
        print(f"Best ECM saved at epoch {epoch + 1} with loss {best_loss:.9f}")
    torch.cuda.empty_cache()
    if (epoch + 1) % 1 == 0:
        print('-' * 89)
        print(f"Epoch {epoch + 1}: Learning Rate: {optimizer.param_groups[-1]['lr']:.6f}, , Loss: {total_loss_epoch  / len(Dataloaders_train):.9f}")
        print('-' * 89)
        print(f"Predicted: {predy[-11:-1].reshape(1,-1)}")
        print(f"Original: {y.unsqueeze(1)[-11:-1].reshape(1,-1)}")
        print(f"Around error: {60*(y.unsqueeze(1)[-11:-1].reshape(1,-1) - predy[-11:-1].reshape(1,-1))}")
    # Save checkpoint
    if (epoch + 1) % 100 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': ECM.module.state_dict(),
            "encoder_state_dict" : ECM.module.encoder.state_dict(),  # Get encoder weights
            "ecoder_state_dict" : ECM.module.decoder.state_dict(),  # Get decoder weights
            'optimizer_state_dict': optimizer.state_dict(),
        }
        # torch.save(checkpoint, os.path.join(models_folder, 
        #                                     f'ECM_checkpoint_epoch_{epoch+1}.pt'))