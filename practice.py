# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/31 20:30'
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv("train.csv")
data_label = ["city","hour","is_workday","weather","temp_1","temp_2","wind"]
data = np.array(raw_data[data_label])
target = np.array(raw_data["y"])

print(train_test_split(data,target,))


# standerScale = StandardScaler()
# standerScale.fit(data)
# data = standerScale.transform(data)
#
# lr = LinearRegression()
# lr.fit(data,target)
# pred = lr.predict(data)
# # test = pd.read_csv("test.csv")
# # test_data = np.array(test[data_label])
# print(mean_squared_error(pred,target))
#
# # test_data = standerScale.transform(test_data)
# # lr.predict(test_data)
#
#
