# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:33:03 2020

@author: lilliloo
"""

#------------------------------------------#
#重回帰分析の数式
#y = b + a_1*x_1 + a_2*x_2 + a_3x_3  
#
#y	目的変数(量的データ)
#x_1〜x_n	複数の説明変数(量的データでも質的データでも可)
#a_1〜a_n	偏回帰係数
#b	定数項(切片)

#------------------------------------------#

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#データの確認
df = pd.read_csv("train.csv")
df.plot(x='datetime', y='y', figsize=(12,5), title='時系列に沿ったお弁当の販売個数推移')

# 1.複数の説明変数を選択
data = df[["temperature","week"]]

# 2. 説明変数の中に質的データが含まれていないか確認
#　weekが質的データ
#　ダミー変数化する（０か１の二値変換させること）

# 3. 質的データをダミー変数化させる
data_new = pd.get_dummies(data)

# 4. 重回帰分析をする

#説明変数
x = pd.get_dummies(df[["temperature","week"]])
#目的変数
y = df["y"]

X = sm.add_constant(x)

#最小二乗法でモデル化
model = smf.OLS(y, X)
result = model.fit()


results = result.summary()
print(results)

#--------------------------------#
#R-squared	決定係数。（1に近いほど精度の高いモデルであることを示す値）
#Adj. R-squared	自由度調整済み決定係数。決定係数は説明変数が増えるほど1に近づく性質があるため、説明変数が多い場合は、決定係数ではなく自由度調整済み決定係数の値を利用。
#AIC	モデルの当てはまり度を示す。小さいほど精度が高い。相対的な値である。
#coef	回帰係数
#std err	二乗誤差
#t	t値。それぞれの説明変数が目的変数に与える影響の大きさを表します。つまり絶対値が大きいほど影響が強いことを意味します。1つの目安としてt値の絶対値が2より小さい場合は統計的にはその説明変数は目的変数に影響しないと判断します。
#p	p値。それぞれの説明変数の係数の有意確率を表します。一般的に、有意確率が有意水準以下(5%を下回っている）ならば、その説明変数は目的変数に対して「関係性がある=有意性が高い」ということを示す。
#[0.025 0.975]	95%信頼区間。


#回帰式
#お弁当の販売個数
#= 113.076 + (-2.5388 × temperature) + (30.8786 × week_月) + 
#(24.4677 × week_火) + (20.5865 × week_水) + (13.1428 × week_木) + 
#(24.0004 × week_金)
#--------------------------------#

pred = result.predict(X)
df["pred"] = pred

df.plot(y=['y','pred'], x='datetime', figsize=(12,5), title='精度44.7％のグラフ')




























