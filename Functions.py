# -*- coding: utf-8 -*-


def labeling(Array):
    """label encoding"""
    # 对出行季节进行Label Encoding
    season_mapping = {'spring': 1, 'summer': 2, 'autumn': 3, 'winter': 4}
    Array['出行季节'] = Array['出行季节'].map(season_mapping)

    # 对出行日期进行Label Encoding
    day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
                   'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    Array['出行日期'] = Array['出行日期'].map(day_mapping)

    # 对出行时段进行Label Encoding
    period_mapping = {'morning peak': 1, 'night peak': 2, 'other time': 3, "nighttime": 4}
    Array['出行时段'] = Array['出行时段'].map(period_mapping)

    # 对车辆类型进行Label Encoding
    vehicle_mapping = {'Sedan': 1, 'SUV': 2, 'Sedan PHEV': 3, 'SUV PHEV': 4}
    Array['车辆类型'] = Array['车辆类型'].map(vehicle_mapping)

    Array.loc[Array['VV'].apply(lambda x: isinstance(x, str)), 'VV'] = 0.1

    columns_to_drop = ["出行时间", "地点", "VIN"]
    Array = Array.drop(columns=columns_to_drop).astype(float)
    Array = Array.fillna(0)

    return Array
