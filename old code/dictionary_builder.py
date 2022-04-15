# -*- coding:utf-8 -*-
"""
@author: ningshixian
@file: test.py
@time: 2019/7/26 17:50
https://www.cnblogs.com/en-heng/p/5848553.html

挖掘行业关键词
TF-IDF正好可用来做关键词的抽取，词TF-IDF值越大，则说明该词为关键词。
过滤常见词的两个办法：
1、用jieba分词对每一条招聘信息做关键词抽取（也是基于TF-IDF），如此能在生成大doc时剔除掉部分常见词。
2、引入max_df，如果词的df超过某一阈值则被词表过滤。
"""

import codecs
import os
import jieba.analyse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

base_path = "./data/会话历史数据明细-行政.txt"
cut_path = "./data/会话历史数据明细-行政-分词.txt"

def segment():
    """word segment"""
    with codecs.open(cut_path, 'w', encoding='utf-8') as wr:
        with codecs.open(base_path, encoding='utf-8') as f:
            for line in f:
                seg_line = jieba.analyse.extract_tags(line, topK=3, withWeight=False, allowPOS=())
                wr.write(' '.join(seg_line))
                wr.write('\n')


def read_doc_list():
    """read segmented docs"""
    doc_list = []
    with codecs.open(cut_path, "r", "utf-8") as fr:
        doc_list.append(fr.read().replace('\n', ''))
    return doc_list


def stopwordslist(filepath):  
    '''创建停用词list'''
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    # stopwords = stopwordslist('../QuestionCluster/data/中文停用词表.txt')  # 这里加载停用词的路径
    return stopwords 


def xx():
    """word segment + read"""
    doc_list = ''
    with codecs.open(base_path, encoding='utf-8') as f:
        for line in f:
            seg_line = jieba.analyse.extract_tags(line, topK=3, withWeight=False, allowPOS=())
            # 去除由英文和字母组成的关键词
            # 
            doc_list += ' '.join(seg_line) + ' '
            # doc_list.append(' '.join(seg_line))
    return [doc_list]


def tfidf_top(doc_list, max_df, topn):
    vectorizer = TfidfVectorizer(max_df=max_df)
    matrix = vectorizer.fit_transform(doc_list)

    print(matrix.shape) # (1284, 676)
    # words = vectorizer.get_feature_names()
    # for i in range(len(dl)):
    #     print('----Document %d----' % (i))
    #     for j in range(len(words)):
    #         if matrix[i,j] > 1e-5:
    #             print( words[j], matrix[i,j])

    feature_dict = {v: k for k, v in vectorizer.vocabulary_.items()}  # index -> feature_name
    top_n_matrix = np.argsort(-matrix.todense())[:, :topn]  # top tf-idf words for each row
    df = pd.DataFrame(np.vectorize(feature_dict.get)(top_n_matrix))  # convert matrix to df
    return df


# segment()
# dl = read_doc_list()
dl = xx()
tdf = tfidf_top(dl, max_df=1, topn=200)
tdf.to_csv("./data/keywords.txt", header=False, encoding='utf-8')

