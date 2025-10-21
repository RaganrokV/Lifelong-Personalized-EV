import pandas as pd
import numpy as np
import torch


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




def preprocess_data(labeled_df):
    # Rename columns
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

    # Convert specific columns to numeric values
    numeric_columns = [
        'Gross Vehicle Weight', 'Max Power', 'Max Torque', 'Official ECR', 'Battery Rated Capacity'
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values (except for 'car_id') with the median of each column
    df.loc[:, df.columns != 'car_id'] = df.loc[:, df.columns != 'car_id'].apply(
        lambda x: x.fillna(x.median()), axis=0
    )

    return df

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
def normalize(test_set):
    
    for col, bounds in max_min_dict.items():
        if col in test_set.columns:  # 确保列在 DataFrame 中
            min_val = bounds['min']
            max_val = bounds['max']     
            # 应用归一化
            test_set[col] = test_set[col].apply(lambda x: 0 if x < min_val else 
                                                1 if x > max_val else 
                                                (x - min_val) / (max_val - min_val))
    return test_set

