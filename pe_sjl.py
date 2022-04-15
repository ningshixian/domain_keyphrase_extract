from nlp_zero import *
import re
import pandas as pd
import codecs
import csv
from tqdm import tqdm


"""分享一次专业领域词汇的无监督挖掘-苏剑林
https://kexue.fm/archives/6540

pd_reader = pd.read_csv("data/All_Knowledge.csv")
pd_reader.fillna("", inplace=True)

lines = []
for row in pd_reader.values:
    if row[0].strip(): lines.append(row[0].strip())
    for sim in row[1].strip().split("###"):
        if sim.strip(): lines.append(sim.strip())

with codecs.open("data/All_Knowledge.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")
"""

import os
import json
tmp = []
g = os.walk(r"data/wiki_zh")  # 遍历wiki_zh文件夹下所有子文件夹的数据文件
for path,dirs,files in tqdm(g):  
    # root 表示当前正在访问的文件夹路径
    # dirs 表示该文件夹下的子目录名list
    # files 表示该文件夹下的文件list
    for file_name in files:
        with codecs.open(os.path.join(path, file_name)) as f:
            for line in f:
                data = json.loads(line)
                tmp.append(data["text"])


class D: # 读取待挖掘的语料
    def __iter__(self):
        with open('data/LZZXKF_knowledge.txt') as f:   # LZZXKF
            for l in f:
                l = l.strip()
                l = re.sub('[^\u4e00-\u9fa5]+', ' ', l)
                yield l


class DO: # 读取苏神爬取的wiki百科语料（相当于平行语料）
    def __iter__(self):
        for l in tmp:
            l = l.strip()
            l = re.sub('[^\u4e00-\u9fa5]+', ' ', l)
            yield l


# 在待挖掘的语料中做新词发现
f = Word_Finder(min_proba=1e-6, min_pmi=0.5)
f.train(D()) # 统计互信息
f.find(D()) # 构建词库

# 导出词表
words = pd.Series(f.words).sort_values(ascending=False)
print(words[:20])
print(words.sum())

# 在通用大语料中做新词发现
fo = Word_Finder(min_proba=1e-6, min_pmi=0.5)
fo.train(DO()) # 统计互信息
fo.find(DO()) # 构建词库

# 导出词表
other_words = pd.Series(fo.words).sort_values(ascending=False)
other_words = other_words / other_words.sum() * words.sum() # 总词频归一化（这样才便于对比）


"""对比两份语料词频，得到特征词。
对比指标是（比赛方语料的词频 + alpha）/（自己语料的词频 + beta）；
alpha和beta的计算参考自 http://www.matrix67.com/blog/archives/5044
"""

WORDS = words.copy()
OTHER_WORDS = other_words.copy()

total_zeros = (WORDS + OTHER_WORDS).fillna(0) * 0
words = WORDS + total_zeros
other_words = OTHER_WORDS + total_zeros
total = words + other_words

alpha = words.sum() / total.sum()

result = (words + total.mean() * alpha) / (total + total.mean())
result = result.sort_values(ascending=False)
idxs = [i for i in result.index if len(i) >= 2] # 排除掉单字词
# print(result)   # xxx    NaN
# print(type(result))     # <class 'pandas.core.series.Series'>
# print("第一列：", list(result.index))

# 导出csv格式
print("导出语料特征词至 result_1.csv")
pd.Series(idxs[:20000]).to_csv('data/result_1.csv', encoding='utf-8', header=None, index=None)



"""
语义筛选
注意到，按照上述方法导出来的词表，顶多算是“语料特征词”，但是还不完全是“电力专业领域词汇”。如果着眼于电力词汇，那么需要对词表进行语义上的筛选。

我的做法是：用导出来的词表对比赛语料进行分词，然后训练一个Word2Vec模型，根据Word2Vec得到的词向量来对词进行聚类。
"""


# nlp zero提供了良好的封装，可以直到导出一个分词器，词表是新词发现得到的词表。
tokenizer = f.export_tokenizer()

class DW:
    def __iter__(self):
        for l in D():
            yield tokenizer.tokenize(l, combine_Aa123=False)


from gensim.models import Word2Vec

word_size = 100
word2vec = Word2Vec(DW(), vector_size=word_size, min_count=2, sg=1, negative=10)


import numpy as np
from multiprocessing.dummy import Queue


def most_similar(word, center_vec=None, neg_vec=None):
    """根据给定词、中心向量和负向量找最相近的词
    """
    vec = word2vec.wv[word] + center_vec - neg_vec
    return word2vec.wv.similar_by_vector(vec, topn=200)


def find_words(start_words, center_words=None, neg_words=None, min_sim=0.6, max_sim=1., alpha=0.25):
    if center_words == None and neg_words == None:
        min_sim = max(min_sim, 0.6)
    center_vec, neg_vec = np.zeros([word_size]), np.zeros([word_size])
    if center_words: # 中心向量是所有中心种子词向量的平均
        _ = 0
        for w in center_words:
            if w in word2vec.wv.vocab:
                center_vec += word2vec.wv[w]
                _ += 1
        if _ > 0:
            center_vec /= _
    if neg_words: # 负向量是所有负种子词向量的平均（本文没有用到它）
        _ = 0
        for w in neg_words:
            if w in word2vec.wv.vocab:
                neg_vec += word2vec.wv[w]
                _ += 1
        if _ > 0:
            neg_vec /= _
    queue_count = 1
    task_count = 0
    cluster = []
    queue = Queue() # 建立队列
    for w in start_words:
        queue.put((0, w))
        if w not in cluster:
            cluster.append(w)
    while not queue.empty():
        idx, word = queue.get()
        queue_count -= 1
        task_count += 1
        sims = most_similar(word, center_vec, neg_vec)
        min_sim_ = min_sim + (max_sim-min_sim) * (1-np.exp(-alpha*idx))
        if task_count % 10 == 0:
            log = '%s in cluster, %s in queue, %s tasks done, %s min_sim'%(len(cluster), queue_count, task_count, min_sim_)
            print(log)
        for i,j in sims:
            if j >= min_sim_:
                if i not in cluster and is_good(i): # is_good是人工写的过滤规则
                    queue.put((idx+1, i))
                    if i not in cluster and is_good(i):
                        cluster.append(i)
                    queue_count += 1
    return cluster


def is_good(w):
    if re.findall('[\u4e00-\u9fa5]', w) \
        and len(w) >= 2\
        and len(w) < 13\
        and not re.findall('[较很越增]|[多少大小长短高低好差]', w)\
        and not '的' in w\
        and not '了' in w\
        and not '这' in w\
        and not '那' in w\
        and not '到' in w\
        and not w[-1] in '为一人给内中后省市局院上所在有与及厂稿下厅部商者从奖出 单店期时点后号费'\
        and not w[0] in '每各该个被其从与及当为'\
        and not w[-2:] in ["期限","详情","规则","电话","时间","时候","信息","客服","商品","号码","费用","范围","价格","人员"]\
        and not w[:2] in ['考虑', '图中', '每个', '出席', '一个', '随着', '不会', '本次', '产生', '查询', '是否', '作者']\
        and not ('博士' in w or '硕士' in w or '研究生' in w)\
        and not (len(set(w)) == 1 and len(w) > 1)\
        and not (w[0] in '一二三四五六七八九十' and len(w) == 2)\
        and re.findall('[^一七厂月二夕气产兰丫田洲户尹尸甲乙日卜几口工旧门目曰石闷匕勺]', w)\
        and not '进一步' in w:
        return True
    else:
        return False


# 种子词，在第一步得到的词表中的前面部分挑一挑即可，不需要特别准
# start_words = ["京东", "代金券", "网易云音乐", "芒果", "沃尔玛超市", "苏宁电子卡", "大唐火车票", "美团外卖", "团油", "网易严选", "团团免税店", "盒马", "物美超市卡", "中石油加油卡", "喜马拉雅", "龙湖团团游", "京东商城", "奈雪的茶", "大麦网", "喜茶", "青苗儿童口腔", "滴滴", "珑珠优选", "好福利", "优酷会员卡", "苹果", "肯德基", "淘票票", "屈臣氏", "天猫超市", "家乐福"]

# 种子词汇获取（利用已有 tag_name）
start_words = []
with codecs.open("data/LZZXKF_tag.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    reader.__next__()
    for row in reader:
        start_words.append(row[0])
        for sim in row[1].split("###"):
            if sim and len(sim) > 1 and not sim.encode("utf-8").isalnum():
                start_words.append(sim)
start_words = list(sorted(set(start_words), key=start_words.index))
start_words = list(set(start_words) & set(word2vec.wv.index_to_key))   # 获得当前的所有词向量表

# 开始挖掘
cluster_words = find_words(start_words, min_sim=0.6, alpha=0.35)

result2 = result[cluster_words].sort_values(ascending=False)
idxs = [i for i in result2.index if is_good(i)]

print("导出领域词至 result_1_2.csv")
pd.Series([i for i in idxs if len(i) > 2][:10000]).to_csv('data/result_1_2.csv', encoding='utf-8', header=None, index=None)
