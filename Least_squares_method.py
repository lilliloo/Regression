# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:52:57 2020

@author: lilliloo
"""


import numpy as np
import matplotlib.pyplot as plt

#説明変数(1次元)
x = np.arange(-3,7,0.5)
#応答変数(説明変数の3次元関数とし、
#正規分布に基づいて乱数的に生成した。
y = 10*np.random.rand()+x * np.random.rand() + 2*np.random.rand()*x**2  +x**3

#描画
plt.scatter(x,y)
plt.show()

#1次式
coef_1 = np.polyfit(x,y,1) #係数
y_pred_1 = coef_1[0]*x+ coef_1[1] #フィッティング関数

#2次式
coef_2 = np.polyfit(x,y,2) 
y_pred_2 = coef_2[0]*x**2+ coef_2[1]*x + coef_2[2] 

#3次式
coef_3 = np.polyfit(x,y,3) 
y_pred_3 = np.poly1d(coef_3)(x) #np.poly1d,求めた係数coef_3を自動で式に当てはめる。

#描画

plt.scatter(x,y,label="raw_data") #元のデータ
plt.plot(x,y_pred_1,label="d=1") #1次式
plt.plot(x,y_pred_2,label="d=2") #2次式
plt.plot(x,y_pred_3,label = "d=3") #3次式
plt.legend(loc="upper left")
plt.title("least square fitting")
plt.show()













