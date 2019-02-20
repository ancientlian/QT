# -*- coding: utf-8 -*-
"""
__title__ = ''
__file__ = '沪深300聚类分析.py' 
__author__ = 'Administrator'
__mtime__ = '2019/2/18'
# 
生命中的孤独时刻只是一个人心灵的安静，
并不是一种难以忍受的负面情绪。
#
"""

import numpy as np
import pandas as pd
import tushare as ts
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import cluster,covariance,manifold

from matplotlib.font_manager import FontProperties




# 设置tushare的token认证
print('ts_version = '+ ts.__version__)
pro = ts.pro_api('18485ae546a437482acd7aa7f48e5b84e6fcf18a12641c2c20c257b7')

#获取沪深300指数的股票名单
hs300_data=ts.get_hs300s()


# 查询获取上交所SSE，深交所SZSE，(港交所HKEX)正常上市交易的股票名单
exc=["SSE","SZSE"]
stock_data=[]
for ex in exc:
    data=pro.query('stock_basic', exchange=ex,list_status='L',fields='ts_code,symbol,name,area,industry,list_date')
    stock_data.append(data)

#获取沪深300成分股中正常上市交易的名单
#将stock_data中上交所和深交所中的交易数据合并
s_name=pd.concat([stock_data[0][["name","ts_code"]],stock_data[1][["name","ts_code"]]],ignore_index=True)

#找出沪深300指在上交所和深交所的交易代码
hs300_data=hs300_data.set_index('name')
print(hs300_data)
s_name=s_name.set_index('name')
# print(s_name)
sdata=pd.merge(hs300_data,s_name,on='name',how='inner')
# print(sdata)
ts_code=sdata['ts_code'].values
# print(ts_code)
n=len(ts_code)

#提取沪深300指2010年01月01到2018年01月01的交易数据，存入d_price中
d_price=[]
names=[]
symbols=[]
for i in range(62,199):
    df = pro.daily(ts_code=ts_code[i],start_date='20181201',end_date='20190101')
    d_price.append(df)
    names.append(sdata[sdata['ts_code']==ts_code[i]].index.tolist())
    symbols.append(ts_code[i])

names=pd.DataFrame(names)
symbols=pd.DataFrame(symbols)

op=[]
cl=[]
for q in d_price:
    op.append(q['open'].values)
    cl.append(q['close'].values)

close_prices=np.vstack([i for i in op])
open_prices=np.vstack([j for j in cl])


# The daily variations of the quotes are what carry most information
variation = close_prices - open_prices

# #############################################################################
# Learn a graphical structure from the correlations
# 稀疏的可逆协方差估计，L1型，cv迭代5次
edge_model = covariance.GraphicalLassoCV(cv=5)

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
# 标准化时间序列，使用相关性而不是协方差的原因是在结构恢复时更高效
X = variation.copy().T
X /= X.std(axis=0)  # 标准差
edge_model.fit(X)

# #############################################################################
# Cluster using affinity propagation
# 使用近邻传播算法构建模型，并训练LassoCV graph
_, labels = cluster.affinity_propagation(edge_model.covariance_)
print('labels = ', labels)
n_labels = labels.max()
print('n_labels = ', n_labels)
print(names)
for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(names[0][labels == i])))

# #############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane
# 可视化低维嵌入

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
# LLE（局部线性嵌入）
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T

# #############################################################################
# Visualization
plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)  # np.triu返回上角矩阵

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.nipy_spectral)

# Plot the edges
# 画出股票连线，颜色的深浅代表两者的相关性
start_idx, end_idx = np.where(non_zero) # 以元组形式返回值为true的坐标
# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero]) # 行列式子变换后的矩阵的非零值
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
# lc.set_linewidths(15 * values)
lc.set_linewidths(10 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
# x,y为表示股票坐标的点的坐标
for index, (name, label, (x, y)) in enumerate(
        zip(names[0], labels, embedding.T)):

    dx = x - embedding[0]  # 某一个点的横坐标于其他所有点的横坐标的差值，长度为56
    dx[index] = 1  # 设第index个值为1，原值为0
    dy = y - embedding[1]  # 某一个点的纵坐标于其他所有点的纵坐标的差值，长度为56
    dy[index] = 1  # 设第index个值为1，原值为0
    this_dx = dx[np.argmin(np.abs(dy))]
    # np.argmin()返回最小值所在的下标,本语句为求出dy绝对值最小值所在的dx坐标
    this_dy = dy[np.argmin(np.abs(dx))]
    # np.argmin()返回最小值所在的下标,本语句为求出dy绝对值最小值所在的dx坐标
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                       alpha=.6),fontproperties=FontProperties(fname='../Fonts/PingFang.ttc'))  # 字体

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.show()

