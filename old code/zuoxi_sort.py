#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import pandas as pd
import codecs
from harvesttext import HarvestText
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import jieba.analyse
import re
import numpy as np
from tqdm import tqdm
import arrow
import pickle
from collections import defaultdict, Counter
from string import punctuation


now = str(arrow.now().date())


def read_excel(f_path, sheet):
    """读取Excel文件"""
    exc = pd.io.excel.ExcelFile(f_path)
    df = pd.read_excel(exc, sheet_name=sheet, dtype=str)
    df.fillna("", inplace=True)
    df_li = df.values.tolist()
    return df_li


def new_word_discover(text, flag):
    """使用HarvestText库的新词发现功能
    写入文件后，进一步人工筛选
    """
    ht = HarvestText()
    new_words_info = ht.word_discover(
        doc=text,
        threshold_seeds=[],
        auto_param=True,
        max_word_len=8,
        min_freq=0.00005,
        min_entropy=1.4,
        min_aggregation=50,
        ent_threshold="both",
        mem_saving=0,
    )
    new_words = new_words_info.index.tolist()
    with codecs.open("results/new_word_" + flag + ".txt", "w", "utf-8") as f:
        for w in new_words:
            f.write(w + "\n")
    return new_words


def tokenize(data, new_word_set):
    """分词
    过滤停用词
    """
    for new_word in new_word_set:
        jieba.add_word(new_word)

    training_data = []
    for sen in data:
        training_data.append(jieba.lcut(sen))
    # training_data = []
    # for sent in data:
    #     sent_cut_new = []
    #     sent_cut = jieba.lcut(sent)
    #     for word in sent_cut:
    #         # 过滤停用词 & 特殊符号替换Mask
    #         if word in stop_words or word == "":
    #             continue
    #         sent_cut_new.append(word)
    #     training_data.append(sent_cut_new)
    return training_data


def tfidf_train(data):
    """
    构建词袋空间VSM(vector space model)：
    1、统计所有文档的词集合，转换成词频矩阵
        对每个文档，都将构建一个向量，向量的值是词语在本文档中出现的次数。
    2、统计每个词语tfidf权值
    3、抽取tf-idf权重矩阵
        元素weight[i][j]表示j词在i类文本中的tf-idf权重
        列是所有文档总共的词的集合
        行是一个向量，向量的值是词语在本文档中的权值
    """
    if os.path.exists("tfidf.pickle"):
        tfidf_weight = pickle.load(open("tfidf.pickle", "rb"))
        word = pickle.load(open("word.pickle", "rb"))
    else:
        # token_pattern参数：匹配模式，默认 r"(?u)\b\w+\b"
        # ?u 放在前面的意思是匹配中对大小写不敏感 \w 匹配字母数字及下划线 \S 匹配任意非空字符
        vectorizer = CountVectorizer(lowercase=True, token_pattern=r"\S+", min_df=1)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(data))
        tfidf_weight = tfidf.toarray()
        # idf_weight = transformer.idf_
        word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        # # 保存（Memory Error）
        # pickle.dump(tfidf_weight, open("tfidf.pickle", "wb"))
        # pickle.dump(word, open("word.pickle", "wb"))
    return tfidf_weight, word


def get_keyphrases(tr4w, keywords, min_occur_num=2):
    """获取关键短语。
    获取 keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num。

    Return:
    关键短语的列表。
    """
    keyword_list = [item.word for item in keywords]  # word_min_len = 1
    keyweight_list = [item.weight for item in keywords]
    keyphrases = set()
    for sentence in tr4w.words_no_filter:
        one = []
        weight = 0
        for word in sentence:
            if word in keyword_list:
                one.append(word)
                weight += keyweight_list[keyword_list.index(word)]
            else:
                if len(one) > 1:
                    keyphrases.add(("".join(one), weight))
                if len(one) == 0:
                    continue
                else:
                    one = []
                    weight = 0
        # 兜底
        if len(one) > 1:
            keyphrases.add(("".join(one), weight))

    keyphrase_list = [phrase for phrase in keyphrases if tr4w.text.count(phrase[0]) >= min_occur_num]
    keyphrase_list.sort(key=lambda x: -x[1])
    return keyphrase_list


if __name__ == "__main__":
    print("# 读取语料...")
    rs_data = read_excel("./坐席语料1.xlsx", 0)
    rs_data = [x[6] for x in rs_data]
    xz_data = read_excel("./坐席语料1.xlsx", 1)
    xz_data = [x[6] for x in xz_data]
    cw_data = read_excel("./坐席语料1.xlsx", 2)
    cw_data = [x[6] for x in cw_data]

    print("++==============================++")
    # pip install textrank4zh
    from textrank4zh import TextRank4Keyword, TextRank4Sentence

    excel_filepath = "results/坐席语料textrank4zh关键词抽取.xlsx"
    writer = pd.ExcelWriter(excel_filepath)

    tr4w = TextRank4Keyword()
    tr4w.analyze(text="\n".join(rs_data), lower=True, window=2)
    print("关键词：")
    kw_dict = {"词": [], "权重": []}
    keywords = tr4w.get_keywords(num=350, word_min_len=2)
    for item in keywords:
        kw_dict["词"].append(item.word)
        kw_dict["权重"].append(item.weight)
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="人事-关键词", index=False)
    print("关键短语：")
    kw_dict = {"短语": [], "权重": []}
    for phrase in get_keyphrases(tr4w, keywords, min_occur_num=1):
        kw_dict["短语"].append(phrase[0])
        kw_dict["权重"].append(phrase[1])
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="人事-关键短语", index=False)

    tr4w = TextRank4Keyword()
    tr4w.analyze(text="\n".join(xz_data), lower=True, window=2)
    print("关键词：")
    kw_dict = {"词": [], "权重": []}
    keywords = tr4w.get_keywords(num=350, word_min_len=2)
    for item in keywords:
        kw_dict["词"].append(item.word)
        kw_dict["权重"].append(item.weight)
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="行政-关键词", index=False)
    print("关键短语：")
    kw_dict = {"短语": [], "权重": []}
    for phrase in get_keyphrases(tr4w, keywords, min_occur_num=1):
        kw_dict["短语"].append(phrase[0])
        kw_dict["权重"].append(phrase[1])
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="行政-关键短语", index=False)

    tr4w = TextRank4Keyword()
    tr4w.analyze(text="\n".join(cw_data), lower=True, window=2)
    print("关键词：")
    kw_dict = {"词": [], "权重": []}
    keywords = tr4w.get_keywords(num=350, word_min_len=2)
    for item in keywords:
        kw_dict["词"].append(item.word)
        kw_dict["权重"].append(item.weight)
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="财务-关键词", index=False)
    print("关键短语：")
    kw_dict = {"短语": [], "权重": []}
    for phrase in get_keyphrases(tr4w, keywords, min_occur_num=1):
        kw_dict["短语"].append(phrase[0])
        kw_dict["权重"].append(phrase[1])
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="财务-关键短语", index=False)

    writer.save()
    print("结果写入完成！")
    print("++==============================++")

    # print("# 新词发现...")
    # new_word_rs = new_word_discover(rs_data, flag="rs")
    # new_word_xz = new_word_discover(xz_data, flag="xz")
    # new_word_cw = new_word_discover(cw_data, flag="cw")

    # print("# 分词...")
    # words_list_rs = tokenize(rs_data, new_word_rs)
    # words_list_xz = tokenize(xz_data, new_word_xz)
    # words_list_cw = tokenize(cw_data, new_word_cw)

    # print("tf-idf关键词抽取...")
    # kw_dict = {}
    # words_list = [" ".join(x) for x in words_list_rs]
    # tfidf_weight, word_dict = tfidf_train(words_list)
    # assert len(tfidf_weight[0])==len(word_dict)
    # for i in range(len(tfidf_weight)):
    #     weights = tfidf_weight[i]
    #     words = words_list_rs[i]
    #     sen_list, word_list = [], []
    #     for w in words:
    #         try:
    #             ind2 = word_dict.index(w)
    #             if float(weights[ind2])>0.3:
    #                 if w not in kw_dict:
    #                     kw_dict[w] = 1
    #                 else:
    #                     kw_dict[w] += 1
    #         except:
    #             # 'bpm1.0' '&' '2.0' '-'
    #             print("\'{}\'被tfidf过滤了？".format(w))
    # kw_dict = sorted(kw_dict.items(), key=lambda x: -x[1])

    # print("# 停用词/标点数字过滤...")
    # stop_words = []
    # with codecs.open("./中文停用词表.txt", "r", "utf-8") as f:
    #     for line in f:
    #         stop_words.append(line.strip("\n"))
    # kw_dict2 = {"词":[], "权重":[]}
    # for item in kw_dict:
    #     if not (item[0].isspace() or item[0].encode("utf-8").isalnum() or item[0] in punctuation or item[0] in stop_words):
    #         kw_dict2["词"].append(item[0])
    #         kw_dict2["权重"].append(item[1])

    # print("将抽取结果写入Excel...")
    # excel_filepath = "results/坐席语料tf-idf关键词抽取.xlsx"
    # writer = pd.ExcelWriter(excel_filepath)
    # df1 = pd.DataFrame(kw_dict2)
    # df1.to_excel(writer, sheet_name="tf-idf", index=False)
    # writer.save()
    # print("结果写入完成！")

    # print("# 词频统计...")
    # res_rs = Counter(words_list_rs)
    # res_xz = Counter(words_list_xz)
    # res_cw = Counter(words_list_cw)

    # print("# 停用词/标点数字过滤...")
    # stop_words = []
    # with codecs.open("./中文停用词表.txt", "r", "utf-8") as f:
    #     for line in f:
    #         stop_words.append(line.strip("\n"))
    # f = lambda item: not (
    #     item[0].isspace() or item[0].encode("utf-8").isalnum() or item[0] in punctuation or item[0] in stop_words
    # )
    # res_rs = sorted(filter(f, res_rs.items()), key=lambda item: item[1], reverse=True)
    # res_xz = sorted(filter(f, res_xz.items()), key=lambda item: item[1], reverse=True)
    # res_cw = sorted(filter(f, res_cw.items()), key=lambda item: item[1], reverse=True)

    # print("# 转换成词典格式...")
    # dict_rs = {"词": [], "词频": []}
    # dict_xz = {"词": [], "词频": []}
    # dict_cw = {"词": [], "词频": []}
    # for item in res_rs:
    #     dict_rs["词"].append(item[0])
    #     dict_rs["词频"].append(item[1])
    # for item in res_xz:
    #     dict_xz["词"].append(item[0])
    #     dict_xz["词频"].append(item[1])
    # for item in res_cw:
    #     dict_cw["词"].append(item[0])
    #     dict_cw["词频"].append(item[1])

    # print("# 将抽取结果写入Excel...")
    # excel_filepath = "results/坐席语料统计结果.xlsx"
    # writer = pd.ExcelWriter(excel_filepath)
    # df1 = pd.DataFrame(dict_rs)
    # df1.to_excel(writer, sheet_name="人事", index=False)
    # df2 = pd.DataFrame(dict_xz)
    # df2.to_excel(writer, sheet_name="行政", index=False)
    # df3 = pd.DataFrame(dict_cw)
    # df3.to_excel(writer, sheet_name="财务", index=False)
    # # 必须运行writer.save()，不然不能输出到本地
    # writer.save()
    # print("结果写入完成！")
