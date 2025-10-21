import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
import torch.utils.data as Data
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

        # residual = encoder_output 
        decoded = encoder_output 

        decoded = self.bn(decoded)      
        decoded = self.activation(decoded)
        decoded= self.dropout(decoded)
        decoded = self.out_fc1(decoded.reshape(B,-1))

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


def fine_tune_model(model, dataloader, criterion, optimizer, scheduler, device, num_epochs):
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


def compute_forward_transfer(EC_True, EC_Pred, dist, chunk_size):
    """
    计算 Forward Transfer (FWT)
    """
    # ECR_ERR = np.abs(EC_True.reshape(-1, 1) - EC_Pred.reshape(-1, 1)) / dist.reshape(-1, 1)
    ECR_ERR = np.abs(EC_True.reshape(-1, 1) - EC_Pred.reshape(-1, 1)) 
    
    # 计算每个块的平均值
    chunk_means = [np.mean(ECR_ERR[i:i + chunk_size]) for i in range(0, len(ECR_ERR), chunk_size)]

    # 计算 chunk_means 的平均差异（差分的均值）
    chunk_fwt = np.mean(np.diff(chunk_means))

    # 使用线性拟合来计算 chunk_means 与索引的关系，从而获得斜率
    x = np.arange(len(chunk_means)).reshape(-1, 1)  # 索引
    y = np.array(chunk_means)  # chunk_means 值

    # model = LinearRegression()
    model = HuberRegressor(epsilon=1.2)
    
    model.fit(x, y)
    slope = model.coef_[0]  # 直接获取斜率

    # 计算斜率乘以 dist 均值
    # dist_mean = np.mean(dist)
    # slope_fwt = slope * dist_mean

    return chunk_fwt, slope


def compute_backward_transfer(mae_history, scale_factor, min_val):
    """
    计算 Backward Transfer (BWT)
    """
    BWT_norm = np.mean(np.array(mae_history)[1:] - mae_history[0])
    return BWT_norm * scale_factor + min_val

def create_dataloaders(ev_data, chunk_size, split_ratio):
    """
    创建训练和测试的 DataLoader。
    
    参数:
    - ev_data: pandas DataFrame，包含数据的 DataFrame
    - batch_size: int, 训练集的 batch size
    - chunk_size: int, 测试集的 batch size
    - split_ratio: float, 数据集拆分比例，默认 0.2

    返回:
    - Dataloaders_train: 训练集的 DataLoader
    - Dataloaders_test: 测试集的 DataLoader
    - dist: 测试集的距离
    """
    
    # === 构造数据 ===
    split_index = int(split_ratio * len(ev_data))  # 根据 split_ratio 划分数据集
    trainX = torch.Tensor(ev_data.iloc[:split_index, 1:].values).float()
    trainY = torch.Tensor(ev_data.iloc[:split_index, 0].values).float()
    testX = torch.Tensor(ev_data.iloc[split_index:, 1:].values).float()
    testY = torch.Tensor(ev_data.iloc[split_index:, 0].values).float()
    dist = ev_data.iloc[split_index:, 2].values * 220  # 距离数据

    # 创建 TensorDataset
    train_dataset = Data.TensorDataset(trainX, trainY)
    test_dataset = Data.TensorDataset(testX, testY)

    # 创建 DataLoader
    Dataloaders_train = Data.DataLoader(train_dataset, batch_size=chunk_size, shuffle=True)
    Dataloaders_test = Data.DataLoader(test_dataset, batch_size=chunk_size, shuffle=False)

    return Dataloaders_train, Dataloaders_test, dist,trainX,trainY 



def reparameterization(model, pretrained_model):
    """
    用预训练参数的（均值 & 标准差）进行初始化
    """
    for (name_new, param_new), (name_old, param_old) in zip(model.named_parameters(), pretrained_model.named_parameters()):
        if param_old.requires_grad:
            with torch.no_grad():
                # 检查参数是否包含 nan，并将其替换为一个很小的值（如 1e-6）
                if torch.isnan(param_old).any():
                    # print(f"Warning: {name_old} contains nan values, replacing with 1e-6.")
                    param_old = torch.where(torch.isnan(param_old), torch.tensor(1e-6, device=param_old.device), param_old)
                
                # 计算均值和标准差
                mean, std = param_old.mean(), param_old.std()
                
                # 检查标准差是否为 nan 或 0，如果是则替换为一个很小的值
                if torch.isnan(std) or std == 0:
                    # print(f"Warning: {name_old} has std={std}, using std=1e-6 instead.")
                    std = torch.tensor(1e-6, device=param_old.device)
                
                # 使用均值和标准差重新初始化参数
                param_new.data = torch.normal(mean, std, size=param_new.shape, device=param_new.device)






def monte_carlo_predict(model, data, mc_samples):
    """
    利用 Monte Carlo Dropout 进行多次前向传播，计算预测均值和标准差。
    
    参数:
      model: 用于预测的模型（确保在训练模式下启用 Dropout）
      data: 输入数据
      mc_samples: 前向传播次数（默认为50）
      
    返回:
      mean_pred: 预测均值
      std_pred: 预测标准差
    """
    model.train()  # 启用 Dropout（注意：这可能会使 BatchNorm 采用训练模式统计）
    preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            preds.append(model(data))
    preds = torch.stack(preds, dim=0)  # 形状: (mc_samples, B, output_dim)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    return mean_pred, std_pred

def compute_prediction_intervals(mean_pred, std_pred, z=1.96):
    """
    计算预测区间
    :param mean_pred: 预测均值 (tensor)
    :param std_pred: 预测标准差 (tensor)
    :param z: 对应置信度的 z 值，默认 95% 置信度 (1.96)
    :return: 下界和上界
    """
    lower_bound = mean_pred - z * std_pred
    upper_bound = mean_pred + z * std_pred
    return lower_bound, upper_bound

def compute_picp_mpiw(lower_bound, upper_bound, true_values):
    """
    计算 PICP 和 MPIW
    :param lower_bound: 预测下界 (numpy array)
    :param upper_bound: 预测上界 (numpy array)
    :param true_values: 真实值 (numpy array)
    :return: PICP, MPIW
    """
    n = len(true_values)
    # Indicator: 1 if true value is within the interval, else 0.
    indicators = np.where((true_values.reshape(-1,1) >= lower_bound.reshape(-1,1)) & (true_values.reshape(-1,1) <= upper_bound.reshape(-1,1)), 1, 0)
    picp = np.sum(indicators) / n
    mpiw = np.mean(upper_bound - lower_bound)
    return picp, mpiw

# # 示例：在评估阶段使用 MC Dropout 得到均值和标准差
# mean_pred, std_pred = monte_carlo_predict(model, input_data, mc_samples=50)
# lower_bound, upper_bound = compute_prediction_intervals(mean_pred, std_pred, z=1.96)

# # 假设 true_values 是你的真实标签，均为 numpy 数组
# picp, mpiw = compute_picp_mpiw(lower_bound.cpu().numpy(), upper_bound.cpu().numpy(), true_values)
# print("PICP: {:.2f}, MPIW: {:.2f}".format(picp, mpiw))