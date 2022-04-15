# -*- coding:utf-8 -*-
"""
@author: ningshixian
@file: main.py
@time: 2019/7/23 15:08
"""
import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import joblib
from sklearn.metrics import calinski_harabasz_score
import string
from zhon.hanzi import punctuation  # 中文标点符号
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号


def stopwordslist(filepath):  
    '''创建停用词list'''
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords 

root = 'data'
# xls_path = '会话历史数据明细-行政.xlsx'
# write_path = '会话历史数据明细-行政.txt'
# cluster_result_path = '行政聚类结果'

xls_path = '会话历史数据明细-财务.xlsx'
write_path = '会话历史数据明细-财务.txt'
cluster_result_path = '财务聚类结果'

# 读取excel文件
io = pd.io.excel.ExcelFile(os.path.join(root, xls_path))
df = pd.read_excel(io, na_values=['NA'], usecols=['用户问题', '回复答案'])
# print(df.head(5))
io.close()

# 忽略重复问题
# 包含<b><br>标签的答案也被视为无答案，bug?
df_li = df.values.tolist()
questions = []
for s_li in df_li:
    q, a = str(s_li[0]).strip().strip('\n'), str(s_li[1]).strip().strip('\n')
    # 去掉句子最后的标点符号
    if q[-1] in string.punctuation or q[-1] in punctuation:
        q = q[:-1]
    if a == '暂无答案' and q and q not in questions:
        questions.append(q)
with open(os.path.join(root, write_path), 'w', encoding='utf-8') as f:
    for line in questions:
        f.write(line + '\n')

print('预处理后的问题总数：{}'.format(len(questions)))

sentence_lens = [0]*90
for line in questions:
    if len(line)>len(sentence_lens):continue
    sentence_lens[len(line)] += 1

from operator import itemgetter
sorted_sens = sorted(enumerate(sentence_lens), key=itemgetter(1), reverse=True)
print('[句子长度, 句子个数] 排序前五：{}'.format(sorted_sens[:10]))
# print(np.argsort(sentence_lens)[::-1])

color = ['red', 'black', 'peru', 'orchid', 'deepskyblue']
plt.xticks(range(0, len(sentence_lens), 10))  # 绘制x刻度标签
plt.bar(range(0, len(sentence_lens)), sentence_lens, color=color)  # 绘制y刻度标签
plt.grid(True, linestyle=':', color='r', alpha=0.6)  # 设置网格刻度
plt.xlabel("句子长度区间")  # 显示横轴标签
plt.ylabel("频数/频率")  # 显示纵轴标签
plt.title("频数/频率分布直方图")  # 显示图标题
plt.savefig('句子长度分布.png')
plt.show()

print('#----------------------------------------#')
print('#                                        #')
print('#               KMeans聚类               #')
print('#                                        #')
print('#--------------------------------------#\n')

# 开启BERT服务，对句子进行向量化
from bert_serving.client import BertClient
bc = BertClient(ip='10.240.4.47', port=5555, port_out=5556, timeout=1200000, check_version=False,
                check_token_info=False)
X = bc.encode(questions)
print('开启BERT服务 finish!')

# t-SNE降维
tsne = TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)  # 转换后的输出

print('开始对数据进行聚类...')
# 设定不同k值以运算
# for n_clusters in range(5,50):
for n_clusters in [9]:
    print('选取{}个聚类簇中心'.format(n_clusters))
    kmeans_path = 'model/kmeans'
    cluster = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1500)
    y = cluster.fit_predict(X_tsne).tolist()
    y = np.array(y)
    joblib.dump(cluster, kmeans_path)
    # # 直接加载已保存模型
    # cluster = joblib.load(kmeans_path)
    # y = cluster.predict(X).tolist()

    chs_values = calinski_harabasz_score(X_tsne, y)
    print('Calinski-Harabasz Score (越大越好):', chs_values)

    result_dict = {}
    for i in range(len(y)):
        if y[i] not in result_dict:
            result_dict[y[i]] = []
        result_dict[y[i]].append(questions[i].strip())

    # 按value长度逆序排序
    result_list = sorted(result_dict.items(), key=lambda d: len(d[1]), reverse=True)
    with open(os.path.join(root, cluster_result_path), 'w', encoding='utf-8') as target:
        for item in result_list:
            target.write('{}\n{}\n\n'.format(item[0], '\n'.join(item[1])))

    # 对结果进行可视化
    cents = cluster.cluster_centers_ #质心
    # cents = tsne.fit_transform(cents)  # t-SNE降维转换后的输出
    labels = cluster.labels_#样本点被分配到的簇的索引
    sse = cluster.inertia_
    
    # 画出聚类结果，每一类用一种颜色
    colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
    # 标识出质心("D"表示形状为菱形，后面表示不同的颜色)
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']

    for i in range(n_clusters):
        index = np.nonzero(labels==i)[0]
        x0 = X_tsne[index,0]
        x1 = X_tsne[index,1]
        y_i = y[index]
        ind = i - len(colors) * (i // len(colors) + 1)   # 颜色的索引
        for j in range(len(x0)):
            plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[ind],\
                    fontdict={'weight': 'bold', 'size': 9})
        # plt.plot(cents[i][0], cents[i][1], mark[i], markersize=12)
        plt.scatter(cents[i,0],cents[i,1], marker='x', color=colors[ind], linewidths=12)

    plt.grid(True, linestyle=':', color='r', alpha=0.6)  # 设置网格刻度
    plt.title("财务数据Kmeans聚类结果")  # 显示图标题
    plt.axis([-60,60,-60,60])
    plt.savefig('kmeans_tsne.png')
    plt.show()

