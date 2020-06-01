# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 03:59:08 2020

@author: lilliloo
"""

#------------------------#

#　重回帰分析の精度を上げるために
# ステップワイズ法
#説明変数の候補がたくさんあれば、
#『とりあえずすべての説明変数を重回帰分析にかけてp値が小さく、
#t値の絶対値が大きいものを探索する』というやり方をステップワイズ法と言います。

#------------------------#


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

df = pd.read_csv("train.csv")

#1.カラムが『datetime』以外のすべての説明変数を入れて重回帰分析を行う

# 前処理
# すべての欠損値を0で置換する
df.fillna(0, inplace=True)
df.loc[df['precipitation'].str.contains('--'), 'precipitation'] = '0'

# モデリング
x = pd.get_dummies(df[['week', 'soldout', 'name', 'kcal', 'remarks', 'event', 'payday', 'weather', 'precipitation', 'temperature']]) # 説明変数
y = df['y'] # 目的変数
# 定数項(y切片)を必要とする線形回帰のモデル式ならば必須
X = sm.add_constant(x)
# 最小二乗法でモデル化
model = smf.OLS(y, X)
result = model.fit()

# 重回帰分析の結果を表示する
print(result.summary())

# ２．p値が全て0.05(場合によって0.1)以下のもの以外を説明変数から取り除く
x = pd.get_dummies(df[['week', 'kcal', 'remarks', 'event', 'temperature']]) # 説明変数
x = x.drop(['remarks_0','remarks_手作りの味','remarks_料理長のこだわりメニュー','remarks_酢豚（28食）、カレー（85食）','remarks_鶏のレモンペッパー焼（50食）、カレー（42食）','week_木','week_金'], axis=1)
y = df['y'] # 目的変数

# 定数項(y切片)を必要とする線形回帰のモデル式ならば必須
X = sm.add_constant(x)

# 最小二乗法でモデル化
model = smf.OLS(y, X)
result = model.fit()

# 重回帰分析の結果を表示する
print(result.summary())


pred = result.predict(X)
df['pred'] = pred

df.plot(y=['y','pred'], x='datetime', figsize=(12,5), title='精度71.6％のグラフ')


## ステップワイズ法　＋　仮説
#具体的にどうやって仮説を思いつくかと言うと、『データの可視化』を行い、傾向や偏り、変曲点、スパイクを目で捉えます
# 1.データの可視化によって仮説を建てる
# 2.仮説から販売個数に影響のありそうなものを見つける
# メニュー名に『カレー』を含む
print(df[df['name'].str.contains('カレー')]['y'].mean())

# メニュー名に『カレー』を含まない
print(df[df['name'].str.match('^(?!.*カレー).*$')]['y'].mean())
# 3. 説明変数に追加する
def curry(x):
    if 'カレー' in x:
        return 1
    else :
        return 0
    
df['curry'] = df['name'].apply(lambda x : curry(x))

#4. 同様に仮説を建て、新たな列として追加
# remarks(特記事項)が『お楽しみメニュー』のときも売れてそう！
def fun(x):
    if x=='お楽しみメニュー':
        return 1
    else :
        return 0

df['fun'] = df['remarks'].apply(lambda x : fun(x))
# datetimeのカラムから年のカラムを作成
df['year'] = df['datetime'].apply(lambda x : x.split('-')[0])
df['year'] = df['year'].astype(np.int)

# datetimeのカラムから月のカラムを作成
df['month'] = df['datetime'].apply(lambda x : x.split('-')[1])
df['month'] = df['month'].astype(np.int)

# 5.重回帰分析を行う
x = pd.get_dummies(df[['year', 'month', 'week', 'kcal', 'fun', 'curry', 'weather', 'temperature']]) # 説明変数
y = df['y'] # 目的変数

# 定数項(y切片)を必要とする線形回帰のモデル式ならば必須
X = sm.add_constant(x)

# 最小二乗法でモデル化
model = smf.OLS(y, X)
result = model.fit()

# 重回帰分析の結果を表示する
print(result.summary())

# 6.精度を視覚化する
pred = result.predict(X)
df['pred'] = pred

df.plot(y=['y','pred'], x='datetime', figsize=(12,5), title='精度77.9％のグラフ')
 
#----------------------------------#

# 一番重要な説明変数の見つけ方

#『ステップワイズ法+仮説』を使って、モデルの精度が高く目的変数に影響がある複数の説明変数を見つけることができました。
#今回得られた説明変数の中で、目的変数に最も影響を与えるのは、その中でも統計量tの絶対値が大きいものだと紹介しましたね。
##しかし「統計量tの絶対値が大きいから」といって、それが一番重要な説明変数かと言われると、そうではありません。具体例を見てみましょう。
#そして、『どれぐらい説明変数を動かせる余地があり、また実際に説明変数をどれくらい動かせる手段があるのか』を考えなければなりません。
#「temperatureとかweekとかweatherが統計量が高いので、これらが重要な説明変数です。」と報告されても、人間である以上天候や曜日を自在に操ることはできないので
#果たしてこの分析結果は正しいと言えるでしょうか？
#どう考えても、言えないですよね。
#この結果の場合の重要な説明変数は、name(お弁当の名前)です。メニューを分析して、
#明らかに好まれているメニューとそうではないメニューを見つけ出し、変えていくことがお弁当の販売個数に対して、
#『最大効果』は低いかもしれませんが、確実にインパクトを与えられますね

#--------------------------------#

#  注意すべきこと

#基本的に説明変数が増えれば増えるほど、重回帰式の精度は高くなると紹介しましたが、それだけがいいことばかりとは限りません。
#当てはめの精度は高いのに、予測精度が低くなることを過学習(オーバーフィッティング)と言います。
#過学習になる原因は、『手持ちデータ』に過剰に適合しすぎたモデルを構築してしまったことです。
#こうなると、いまある『検証用データには当てはまりが良い』が『予測したい新しいデータに回帰式を当てはめると、
#当てはまりが悪くなる』といった減少が起きてしまいます。
#過学習を回避するためには一般的に次に紹介する『クロスバリデーション法』を用います。
#『回帰式を求める分析用のデータ』と、『その当てはまりの良さを確認するためのデータ』の2パターンを用意します。
#今回、重回帰分析用に使用したデータセットには、
#回帰式を求める『train.csv』と当てはまりの良さを確認する『test.csv』の2つが用意されているので、test.csvを使います。

## 多重共線性（マルチコ）
#多重共線性とは、説明変数間で非常に強い相関があることを指し、この値が大きいと回帰係数の分散が大きくなり、モデルの予測結果が悪くなることが知られています。
#ただし、重回帰分析を行う目的が『因果関係の洞察』ではなく、『予測』であれば、気にしなくて大丈夫です。
##summary()の結果でいう、Cond. No.が多重共線性をチェックする指標になります。
#ただし、重回帰分析を行う目的が『因果関係の洞察』ではなく、『予測』であれば、気にしなくて大丈夫です。
#
#summary()の結果でいう、Cond. No.が多重共線性をチェックする指標になります。
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

num_cols = model.exog.shape[1] # 説明変数の列数
vifs = [vif(model.exog, i) for i in range(0, num_cols)]

pd.DataFrame(vifs, index=model.exog_names, columns=['VIF'])
#
#一般的にVIFの値が10(公式のリファレンスでは、5)を超えると、依存関係が強いため、適切な重回帰分析ができないと言われています。
#
#今回でいうと、ダミー変数化で作成した『week』の列のVIF値がすべて『inf』となっており、依存関係が非常に強いです。
#
#繰り返しになりますが、重回帰分析の目的が『因果関係の洞察』であれば説明変数から除外したほうが無難であり、『予測』が目的であれば除外しなくても大丈夫です。









































