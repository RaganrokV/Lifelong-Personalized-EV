#%% -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
from My_utils.evaluation_scheme import evaluation
warnings.filterwarnings("ignore")
import torch
import math
import torch.nn as nn
import time
import random
from functions import labeling, preprocess_data,normalize
from models import TransformerEncoder,EnergyPredictionHead,GeneralPredictionModel
from models import fine_tune_model,compute_forward_transfer,compute_backward_transfer
from models import  create_dataloaders
import torch.nn.functional as F
#%% 生成模型定义 
class SymmetricVAE(nn.Module):
    """
    对称 VAE 模型，用于生成伪样本。
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=None):
        super(SymmetricVAE, self).__init__()
        self.input_dim = input_dim
        hidden_dims = hidden_dims or [256, 128]

        # 编码器
        encoder_layers = []
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            input_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # 解码器
        hidden_dims.reverse()
        decoder_layers = []
        for h_dim in hidden_dims:
            decoder_layers.append(nn.Linear(latent_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            latent_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[-1], self.input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_divergence

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
        self.output_layer = nn.Linear(d_model, 1)  # 输出单个特征值

        self.bn = nn.BatchNorm1d(55)
        self.dropout = nn.Dropout(0.1)  # 添加Dropout层

    def forward(self, z, src):
        """
        z: (B, d_model) 隐变量
        src: (B, F) 输入特征，用作条件
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


# === VAE 模型 ===
class TransformerVAE(nn.Module):
    def __init__(self, encoder, decoder, d_model=512):
        super(TransformerVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model

    def reparameterize(self, mu, logvar):
        """通过重参数化采样隐变量 z"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src):
        """
        src: (B, F) 输入特征
        """
        # 编码阶段
        _, mu, logvar = self.encoder(src)
        z = self.reparameterize(mu, logvar)

        # 解码阶段
        reconstructed = self.decoder(z, src)
        
        return reconstructed, mu, logvar
    
class BIR:
    """
    Brain-inspired replay
    """
    def __init__(self, main_model, generator, latent_dim, device):
        self.main_model = main_model
        self.generator = generator
        self.latent_dim = latent_dim
        self.device = device

    def update_generator(self, svae, x, y, optimizer, scheduler, device, epochs=5):
        svae.to(device).train()
        x, y = x.to(device), y.to(device)
        for epoch in range(epochs):
            combined = torch.cat((x, y.unsqueeze(1)), dim=1)
            optimizer.zero_grad()
            recon, mu, logvar = svae(combined)
            loss = svae.loss_function(recon, combined, mu, logvar)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def generate_replay_data(self,generator, num_samples, latent_dim, device):
        generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim).to(device)
            generated_data = generator.decode(z)
        return generated_data

    
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
ECM = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)


ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/best_ECM.pt')
# ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/ECM_checkpoint_epoch_100.pt')

total_veh=[]
counter = 0  # 初始化计数器
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
    optimizer = torch.optim.AdamW(ECM.parameters(), lr=1e-4, 
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', patience=2, factor=0.5, verbose=True)

    # ==== 微调阶段 ======
    fine_tune_model(ECM, Dataloaders_train, criterion, optimizer, 
                    scheduler, device,num_epochs=3)

    del optimizer, scheduler

    # 微调生成模型

    
    encoder = TransformerEncoder(normalized_attribution_ids.to(device), d_model=d_model, max_len=max_len, 
                                 num_layers=num_layers, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    decoder = TransformerDecoder(d_model=d_model,  max_len=max_len, num_layers=num_layers, nhead=nhead,
                                  dim_feedforward=dim_feedforward, dropout=dropout).to(device)
    EVVAE = TransformerVAE(encoder, decoder, d_model=d_model)

    enc_para = torch.load('/home/ps/haichao/1-lifelong_learning/Model/model_checkpoint_epoch_2000.pt')
    EVVAE.load_state_dict(enc_para)



    # latent_dim = 10
    # svae = SymmetricVAE(input_dim=Normalized_array.shape[1], latent_dim=latent_dim)
    # generator_optimizer = torch.optim.Adam(svae.parameters(), lr=1e-3)
    # generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(generator_optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    # fine_generator(svae, Dataloaders_train, generator_optimizer, generator_scheduler, device)

    # 在线学习阶段
    bir = BIR(ECM, EVVAE, device, feat_dim)


    epochs=5
    online_optimizer = torch.optim.AdamW(ECM.parameters(), lr=1e-6, 
                                         betas=(0.9, 0.999), weight_decay=1e-1)
    online_scheduler = torch.optim.lr_scheduler.StepLR(online_optimizer, 
                             1, gamma=0.5, last_epoch=-1, verbose=True)


    preds, reals, old_task_mae = [], [], []

    infer_time = 0.0  # 
    train_time = 0.0 
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

        # 生成伪样本
        pseudo_samples = bir.generate_replay_data(EVVAE, max(len(x), 32), feat_dim, device)

        # 计算损失并优化
        ECM.train()
        # svae.train()
        start_train_time = time.time()  # 开始计时
        # 当前任务的前向传播与训练
        chunk_loss = 0.0
        ECM.train()
        for epoch in range(epochs):
            # online_optimizer.zero_grad()
            current_loss = criterion(ECM(x), y.unsqueeze(1))
            pseudo_loss = criterion(ECM(pseudo_samples[:, 1:]), pseudo_samples[:, 0])
            distillation_loss = F.mse_loss(
                F.softmax(ECM(pseudo_samples[:, 1:]) / 2, dim=0),
                F.softmax(ECM(pseudo_samples[:, 1:]) / 2, dim=0)
            )
            alpha = 1 / (step + 1)
            beta = 1 - alpha
            loss = alpha * current_loss + beta * pseudo_loss + 0.1 * distillation_loss
            loss.backward()
            online_optimizer.step()
            chunk_loss += loss.item()

        print(f"Chunk: {step + 1}/{len(Dataloaders_test)}, Epoch Loss: {chunk_loss:.4f}")
        online_scheduler.step()

        # 更新生成模型
        # bir.update_generator(svae, x, y, generator_optimizer, generator_scheduler, device)

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


    print(f"Direct Measure: {DirectMeasure}, FWT: {FWT}, BWT: {BWT}")
    
    break

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



    