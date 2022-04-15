import pandas as pd
import codecs
from tqdm import tqdm
from harvesttext import HarvestText
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from sklearn.metrics.pairwise import cosine_similarity
import jionlp as jio
from pyhanlp import *
from hashlib import md5
from LAC import LAC
import csv
import warnings

warnings.filterwarnings('ignore')

"""
对多种关键词/短语抽取库进行测试
"""


def encrypt_md5(s):
    # 创建md5对象
    new_md5 = md5()
    # 这里必须用encode()函数对字符串进行编码，不然会报 TypeError: Unicode-objects must be encoded before hashing
    new_md5.update(s.encode(encoding="utf-8"))
    # 加密
    return str(new_md5.hexdigest())


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
    问题：新词发现，很多实体词被忽略了....
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
    return new_words


def get_keyphrases(tr4w, keywords, min_occur_num=2):
    """获取关键短语。
    获取 keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num。

    Return:
    关键短语的列表。
    """
    keyword_list = [item.word for item in keywords]  # word_min_len = 1
    keyweight_list = [item.weight for item in keywords]
    keyphrases = set()
    # for words in tr4w.words_no_filter:
    #     print('/'.join(words))
    for words in tr4w.words_no_filter:
        one = []
        weight = 0
        for word in words:
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

    keyphrase_list = [
        phrase for phrase in keyphrases if tr4w.text.count(phrase[0]) >= min_occur_num
    ]
    keyphrase_list.sort(key=lambda x: -x[1])
    return keyphrase_list


def kw_extract_textrank4zh(text_list, writer):
    tr4w = TextRank4Keyword()
    kw_dict = {"词": [], "权重": []}
    kp_dict = {"短语": [], "权重": []}

    for text in text_list:
        tr4w.analyze(text, window=2)  # , lower=True
        # print("关键词：")
        keywords = tr4w.get_keywords(num=50, word_min_len=2)
        for phrase in get_keyphrases(tr4w, keywords, min_occur_num=1):
            kw_dict["词"].append(phrase[0])
            kw_dict["权重"].append(phrase[1])
        # print("关键短语：")
        # for phrase in get_keyphrases(tr4w, keywords, min_occur_num=1):
        #     kp_dict["短语"].append(phrase[0])
        #     kp_dict["权重"].append(phrase[1])
        keyphrases = tr4w.get_keyphrases(keywords_num=20, min_occur_num=1)
        # print(keyphrases)
        for phrase in keyphrases:
            kp_dict["短语"].append(phrase)
            kp_dict["权重"].append(1)

    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="关键词", index=False)
    df1 = pd.DataFrame(kp_dict)
    df1.to_excel(writer, sheet_name="关键短语", index=False)


def kw_extract_jieba(text_list, writer):
    import jieba

    for text in text_list:
        #textrank
        keywords_textrank = jieba.analyse.textrank(text)
        print(keywords_textrank)

        #tf-idf
        keywords_tfidf = jieba.analyse.extract_tags(text, withWeight=True)
        print(keywords_tfidf)


def kw_extract_harvest(text_list, writer):
    ht = HarvestText()
    kw_dict = {"词": [], "权重": []}

    for text in text_list:
        kwds = ht.extract_keywords(text, 5, method="jieba_tfidf")
        # kwds = ht.extract_keywords(text, 5, method="textrank")
        for item in kwds:
            kw_dict["词"].append(item)
            kw_dict["权重"].append(1)
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="关键词", index=False)


def kw_extract_jionlp(text_list, writer):
    kw_dict = {"词": [], "权重": []}
    for text in text_list:
        key_phrases = jio.keyphrase.extract_keyphrase(text)
        # print(key_phrases)
        for item in key_phrases:
            kw_dict["词"].append(item)
            kw_dict["权重"].append(1)
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="关键短语", index=False)


# def kw_extract_lac(text_list, writer):
#     """
#     rank_result = [
#         ['LAC', '是', '个', '优秀', '的', '分词', '工具'],
#         [nz, v, q, a, u, n, n],
#         [3, 0, 0, 2, 0, 3, 1]
#     ]
#     """
#     from LAC import LAC
#     lac = LAC(mode="rank") # 词语重要性
#     kw_dict = {"词": [], "权重": []}

#     rank_result = lac.run(text_list)
#     print(rank_result)
#     # for item in rank_result:
#     #     res_token = item[0]
#     #     res_pos = item[1]
#     #     res_rank = item[2]
#     #     for i,rank in enumerate(res_rank):
#     #         if rank==3:
#     #             kw_dict["词"].append(res_token[i])
#     #             kw_dict["权重"].append(rank)
#     # df1 = pd.DataFrame(kw_dict)
#     # df1.to_excel(writer, sheet_name="关键短语", index=False)


def kw_extract_pyhanlp(text_list, writer):
    """
    基于依存句法分析的关键短语抽取算法实战
    https://cloud.tencent.com/developer/article/1605278
    """
    kw_dict = {"词": []}
    CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")

    words = {"马上修"}  # 自定义词典
    for text in tqdm(text_list):
        entities = jio.keyphrase.extract_keyphrase(text, top_k=2)
        for e in entities:  # 新词加入用户词典？有点问题
            words.add(e)
    for w in words:
        CustomDictionary.insert(w)

    for text in tqdm(text_list):
        key_phrase = []
        tree = HanLP.parseDependency(text)
        # print(tree)
        for word in tree.iterator():  # 依存树，将具有定中关系的相邻词组抽取出来
            # print("%d,%s-->%s-->%d,%s" % (word.ID, word.LEMMA, word.DEPREL, word.HEAD.ID, word.HEAD.LEMMA))
            if word.DEPREL == "定中关系" and "n" in word.POSTAG and "n" in word.HEAD.POSTAG:
                kw_dict["词"].append(word.LEMMA + word.HEAD.LEMMA)
        # print("key_phrase: ", key_phrase)   # 挺靠谱的
        # print('\n')
    kw_dict["词"] = sorted(set(kw_dict["词"]), key=kw_dict["词"].index)
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="关键短语", index=False)


def kw_extract_jd(text_list, writer):
    """京东 AI-NER 文档
    尝试京东 AI 的 NER api 抽取公司/产品名！！--不满足需求
    https://aidoc.jd.com/thirdDocs/7/5f2f7e09db264d07aac8b973b7efd682-657.html
    https://neuhub.jd.com/purchased/api/list
    """
    import requests
    import time
    import hashlib

    timestamp = int(time.time() * 1000)  # 时间戳
    sign = encrypt_md5("ac4ac73f162dbf6c5d38ae1055ff6650" + str(timestamp))

    kw_dict = {"词": []}
    for text in tqdm(text_list):
        url = f"https://aiapi.jd.com/jdai/nlp_ner?Content-Type=application/json&appkey=e5f078964d81c90527dc879f0551dc02&timestamp={timestamp}&sign={sign}"
        res = requests.post(
            url=url,
            json={"content": text},
        )
        data = res.json().get("result")
        # print(data.get("result"))
        if data:
            for item in data.get("data")[0].get("results"):
                if item["type"] in ["公司", "产品"]:
                    kw_dict["词"].append(item["entity"])

    kw_dict["词"] = sorted(set(kw_dict["词"]), key=kw_dict["词"].index)
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="关键短语", index=False)


def kw_extract_corenlp(text_list, writer):
    """sudo python kw_extract.py
    https://github.com/Lynten/stanford-corenlp
    跑步起来啊....
    """
    from stanfordcorenlp import StanfordCoreNLP

    nlp = StanfordCoreNLP(
        r"/Users/admin/Desktop/longfor.project/corpus/stanford-corenlp",
        lang="zh",
        memory="4g",
    )
    sentence = "我的家在东北，松花江上啊啊"
    print("Tokenize:", nlp.word_tokenize(sentence))
    print("Part of Speech:", nlp.pos_tag(sentence))
    print("Named Entities:", nlp.ner(sentence))
    print("Constituency Parsing:", nlp.parse(sentence))
    print("Dependency Parsing:", nlp.dependency_parse(sentence))
    nlp.close()


def kw_extract_smoothnlp(text_list, writer):
    """
    https://github.com/smoothnlp/SmoothNLP/tree/master/tutorials/%E6%96%B0%E8%AF%8D%E5%8F%91%E7%8E%B0
    乱七八糟的....
    """
    from smoothnlp.algorithm.phrase import extract_phrase
    corpus = text_list
    top_k = 0.2  # 取前k个new words或前k%的new words
    chunk_size = 1000000   # 用chunksize分块大小来读取文件
    min_n = 2   # 抽取ngram及以上
    max_n = 5    # 抽取ngram及以下
    min_freq = 1    # 抽取目标的最低词频
    
    key_phrases = extract_phrase(corpus,top_k,chunk_size,min_n,max_n,min_freq)

    kw_dict = {"词": [], "权重": []}
    for item in key_phrases:
        kw_dict["词"].append(item)
        kw_dict["权重"].append(1)
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="关键短语", index=False)


if __name__ == "__main__":

    domain = "LZZXKF"   # ZHJKF
    root = "data/"
    text_list = [
        "如何抵扣App store的充值卡?",
        "如何抵扣并使用芒果TV会员卡？",
        "网易云音乐VIP卡是否可以开发票？",
        "兑付的Keep会员卡如何使用",
        "花加鲜花卡如何使用",
        "喜茶礼品卡如何使用",
        "充值失败什么时候能退回呢？",
        "为什么我一直看不到预定页面?",
        "装修+软装如何申请售后？",
        "一珑珠等于几京豆",
        "为什么我在充值的时候，提示我“运营商维护中，请稍后重试”？",
        "手机充值后能取消吗？"
    ]

    # domain = "LZZXKF"   # ZHJKF
    # root = "data/"
    # res = read_excel(root + domain + "知识.xlsx", 0)
    # text_list = []
    # for item in res:
    #     pri, sim = item[0], item[1]
    #     if pri:
    #         text_list.append(pri)
    #     if sim:
    #         text_list.extend(sim.split("###"))
    # print(len(text_list))
    # text_list = list(set(text_list))[:]  # 1000
    # text_list = list(filter(lambda x: bool(x), text_list))  # 过滤空字符串

    # 预处理
    # 获取停用词
    data = open( root + "中文停用词表.txt", "r", encoding="utf-8").read()
    stopword = data.split("\n")
    lac = LAC(mode="seg")
    sen_list = lac.run(text_list)
    text_list = []
    for sen in sen_list:
        text_list.append("".join([w for w in sen if w not in stopword]))
    data, lac, sen_list = None, None, None

    excel_filepath = root + domain + "关键词抽取结果.xlsx"
    writer = pd.ExcelWriter(excel_filepath)

    # 执行关键词/短语抽取
    method = [
        "textrank4zh", "harvest", "jionlp", "lac", 
        "pyhanlp", "jd", "corenlp", "smoothnlp"
    ][-1]
    eval("kw_extract_"+method)(text_list, writer)

    writer.save()
    print("结果写入完成！")
    print("++==============================++")
