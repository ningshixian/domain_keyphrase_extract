# -*- coding:utf-8 -*-
"""
@author: 苏剑林
@file: dictionaryBuilder.py
@time: 2019/7/26 10:19
https://kexue.fm/archives/6540
"""
import pandas as pd
import os
import re
import jieba
import string
from tqdm import tqdm
from zhon.hanzi import punctuation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nlp_zero import *
from gensim.models import Word2Vec
import numpy as np
from multiprocessing.dummy import Queue


class D: # 读取比赛方所给语料
    def __iter__(self):
        with open('data/会话历史数据明细-行政.txt', encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                l = re.sub(u'[^\u4e00-\u9fa5]+', ' ', l)
                yield l


class DO: # 读取自己的语料（相当于平行语料）
    def __iter__(self):
        path = "data/wiki_zh"  # 文件夹目录
        base_path = "./data/会话历史数据明细-行政.txt"
        s = []
        with open(base_path, encoding='utf-8') as f:
            for line in f:
                s.append(line.strip('\n'))
        # dirs = os.listdir(path)  # 得到文件夹下的所有文件名称
        # for dir in tqdm(dirs):  # 遍历文件夹
        #     files = os.listdir(os.path.join(path, dir))  # 得到文件夹下的所有文件名称
        #     s = []
        #     for file in files:
        #         if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        #             # f = open(path + "/" + dir + "/" + file);  # 打开文件
        #             data = pd.read_json(path_or_buf=path + "/" + dir + "/" + file, orient='records', encoding='utf-8', lines=True)
        #             s.extend(data['text'])
        for l in tqdm(s[:1000]):
            l = l.strip()
            l = re.sub(u'[^\u4e00-\u9fa5]+', ' ', l)
            yield l


def stopwordslist(filepath):
    '''创建停用词list'''
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# # 进行句子的切分
# def cut(line):
#     result = []
#     line = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', line)[0]    # 去掉无意义字符
#     line_splited = jieba.cut(line.strip().strip('\n'))
#     # tags = jieba.analyse.extract_tags(line.strip().strip('\n'), topK=3) # 获取关键词
#     for item in line_splited:
#         if item not in string.punctuation and item not in punctuation and item not in stopwords:
#             result.append(item)
#     return result


# root = 'data'
# xls_path = '会话历史数据明细-行政.xlsx'
#
# # 读取excel文件
# io = pd.io.excel.ExcelFile(os.path.join(root, xls_path))
# df = pd.read_excel(io, na_values=['NA'], usecols=['用户问题'])
# df_li = df.values.tolist()
# io.close()
#
# word_list = []
# corpus = []
# stopwords = stopwordslist('../QuestionCluster/data/中文停用词表.txt')  # 这里加载停用词的路径
# for line in df_li:
#     words = cut(line[0])
#     if words:
#         corpus.append(' '.join(words))
#     word_list += words
# # word_list = list(set(word_list))
#
# # 统计词频并按照词频由大到小排序，取top500
# cnt = pd.Series(word_list).value_counts()
# cnt.to_csv("data/cnt.csv")
# print(cnt.head(500).to_dict())


tokenizer = jieba.lcut
f = Template_Finder(tokenizer, window=3)
f.train(D())  # 统计互信息
f.find(D())  # 构建词库

# 导出新词发现得到的词表
words = pd.Series(f.words).sort_values(ascending=False)
# print(words)

# 在自己的语料中做新词发现
tokenizer = jieba.lcut
fo = Template_Finder(tokenizer, window=3)
fo.train(DO())  # 统计互信息
fo.find(DO())  # 构建词库

# 导出词表
other_words = pd.Series(fo.words).sort_values(ascending=False)
other_words = other_words / other_words.sum() * words.sum() # 总词频归一化（这样才便于对比）

"""对比两份语料词频，得到特征词。
对比指标是（比赛方语料的词频 + alpha）/（自己语料的词频 + beta）；
alpha和beta的计算参考自 http://www.matrix67.com/blog/archives/5044
"""

a = words.copy()
b = other_words.copy()

total_zeros = (a+b).fillna(0)*0
words = a+total_zeros
other_words = b+total_zeros
total = words+other_words

alpha = words.sum() / total.sum()

result = (words+total.mean()*alpha) / (total+total.mean())
result = result.sort_values(ascending=False)
idxs = [i for i in result.index if len(i)>=2]

# 导出csv格式
pd.Series(idxs[:20000]).to_csv('data/cnt_1.csv',encoding='utf-8')


class DW:
    def __iter__(self):
        for l in D():
            yield tokenizer(l)

word_size = 100
word2vec = Word2Vec(DW(), size=word_size, min_count=2, sg=1, negative=10)


class DW:
    def __iter__(self):
        for l in D():
            yield tokenizer(l)

word_size = 100
word2vec = Word2Vec(DW(), size=word_size, min_count=2, sg=1, negative=10)


def most_similar(word, center_vec=None, neg_vec=None):
    """根据给定词、中心向量和负向量找最相近的词
    """
    vec = word2vec[word] + center_vec - neg_vec
    return word2vec.similar_by_vector(vec, topn=200)


def find_words(start_words, center_words=None, neg_words=None, min_sim=0.6, max_sim=1., alpha=0.25):
    if center_words == None and neg_words == None:
        min_sim = max(min_sim, 0.6)
    center_vec, neg_vec = np.zeros([word_size]), np.zeros([word_size])
    if center_words: # 中心向量是所有种子词向量的平均
        _ = 0
        for w in center_words:
            if w in word2vec.wv.vocab:
                center_vec += word2vec[w]
                _ += 1
        if _ > 0:
            center_vec /= _
    if neg_words: # 负向量是所有负种子词向量的平均（本文没有用到它）
        _ = 0
        for w in neg_words:
            if w in word2vec.wv.vocab:
                neg_vec += word2vec[w]
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
                if i not in cluster: # is_good是人工写的过滤规则
                    queue.put((idx+1, i))
                    if i not in cluster:
                        cluster.append(i)
                    queue_count += 1
    return cluster


# 种子词，在第一步得到的词表中的前面部分挑一挑即可，不需要特别准
start_words = [u'审批', u'发票', u'报销', u'出差', u'差旅', u'滴滴', u'入职', u'车补', u'订票', u'补贴', u'办公用品', u'票据', u'工卡', u'易旅']

cluster_words = find_words(start_words, min_sim=0.6, alpha=0.35)

result2 = result[cluster_words].sort_values(ascending=False)
idxs = [i for i in result2.index]

pd.Series([i for i in idxs if len(i) > 2][:10000]).to_csv('data/cnt_2.csv', encoding='utf-8')

