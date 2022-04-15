import re
import pandas as pd
import codecs
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import jionlp as jio
from LAC import LAC
import csv
import warnings
import collections
from kw_extract_test import read_excel

warnings.filterwarnings('ignore')

"""基于关键短语+依存分析的无监督领域短语挖掘
https://www.yuque.com/ningshixian/pz10h0/psdsng#71162b64

基于匹配-过滤的方法，挖掘行业关键词词库，步骤如下：
1、抽取关键短语，作为自定义词典
2、LAC词语重要性分析
3、DDParser依存分析：
    1、将其中的定中关系的词进行组合（依据词性进行规则过滤）
    2、若上述条件未成功抽取，取第一个出现的名词
4、训练一个关键词质量分类模型，去判断这个ngram是否可能作为一个关键成分短语（待完成）
5、人工过滤 无关的词...

select primary_question,similar_question 
from oc_knowledge_management 
where base_code 
in (select base_code from oc_knowledge_base_flow where flow_code="LZZXKF")
"""


def phrase_extract_textrank4zh(text_list, writer):
    tr4w = TextRank4Keyword()
    kw_dict = {"词": [], "权重": []}
    kp_dict = {"短语": [], "权重": []}

    for text in text_list:
        keyphrases = tr4w.get_keyphrases(keywords_num=20, min_occur_num=1)
        # print(keyphrases)
        for phrase in keyphrases:
            kp_dict["短语"].append(phrase)
            kp_dict["权重"].append(1)

    df1 = pd.DataFrame(kp_dict)
    df1.to_excel(writer, sheet_name="关键短语", index=False)


def request_sbert(url, sen_list):
    """请求sbert服务"""
    import json
    import requests

    headers = {"Content-Type": "application/json"}
    d = json.dumps({"text": sen_list})
    res = requests.post(url, data=d, headers=headers, timeout=5)  # 默认超时时间5s
    sen_embeddings = res.json().get("sbert_vec")
    return sen_embeddings


def phrase_extract_ddparse(text_list, writer):
    """https://github.com/baidu/lac"""
    print("=====len(text_list)=====", len(text_list))

    # # 种子词汇获取（利用已有 tag_name）
    # zhongzi = []
    # with codecs.open(root + flow_code + "_tag.csv", "r", encoding="utf-8") as f:
    #     reader = csv.reader(f)
    #     reader.__next__()
    #     for row in reader:
    #         zhongzi.append(row[0])
    #         for sim in row[1].split("###"):
    #             if sim and len(sim) > 1 and not sim.encode("utf-8").isalnum():
    #                 zhongzi.append(sim)
    # zhongzi = list(sorted(set(zhongzi), key=zhongzi.index))
    # zhongzi.append("喜茶")
    zhongzi = util.zhongzi[:]

    print("抽取关键短语，加入自定义词典（避免被分词）")
    for text in text_list:
        # func_word_num: 允许短语中出现的虚词个数，strict_pos 为 True 时无效
        # stop_word_num: 允许短语中出现的停用词个数，strict_pos 为 True 时无效
        # strict_pos: (bool) 为 True 时仅允许名词短语出现
        # without_person_name: (bool) 决定是否剔除短语中的人名
        # without_location_name: (bool) 决定是否剔除短语中的地名
        keyphrases = jio.keyphrase.extract_keyphrase(
            text,
            func_word_num=0,
            stop_word_num=0,
            strict_pos=True,
            without_person_name=True,
            top_k=2,
        )  # top_k=1
        # print(keyphrases)
        zhongzi.extend(keyphrases)
    zhongzi = list(map(lambda x: x + "/n", set(zhongzi)))

    # 用于词语重要性分析
    lac = LAC(mode="rank")
    for w in zhongzi:
        lac.add_word(w)

    # 用于依存分析
    from ddparser import DDParser
    ddp = DDParser(buckets=True, use_pos=True)

    print("主流程开始：依存分析 → ATT 邻接词组合 → 重要性分析 → 打分")
    kw_dict = {"词": [], "权重": []}
    for j in tqdm(range(len(text_list))):
        text = text_list[j]
        tree = ddp.parse(text, zhongzi)  # [ {word:[], postag:[], head:[], deprel:[]}, {}, ...]
        if tree: tree = tree[0]  
        else: continue  # TypeError: 'NoneType' object is not subscriptable
        doc_embedding = request_sbert("http://10.231.135.106:8096/sbert", text)

        # 找出所有“定中关系”的依存对
        tmp_len = len(kw_dict["词"])
        if "ATT" in tree["deprel"]:  # 定中关系
            w_att_dict = collections.OrderedDict()  # 有序字典
            idx_att = [i for i,x in enumerate(tree["deprel"]) if x=='ATT']  # 查找 att 的多个索引
            for idx in idx_att:
                idx_h = tree["head"][idx] - 1  # 父节点索引，0代表root
                if (
                    # (
                    #     tree["word"][i] + "/n" in zhongzi
                    #     or tree["word"][idx_h] + "/n" in zhongzi
                    # )
                    tree["postag"][idx] in ["n", "nz", "nw", "ORG"]
                    and tree["postag"][idx_h] in ["n", "nz", "nw", "ORG"]
                ):
                    w_att_dict[tree["word"][idx]] = tree["word"][idx_h]

            # 根据 att 路径组合 phrase
            for k,v in w_att_dict.items():
                phrase = k + v
                # 继续组合可能的连续短语，如：APP + Store + 充值卡 → APP Store 充值卡
                visited = []    # 防止陷入死循环，如：w_att_dict=[('团', '团'), ('59', '团')]
                while v in w_att_dict and v not in visited:
                    v = w_att_dict[v]
                    phrase += v
                    visited.append(v)
                    
                if phrase not in kw_dict["词"] and is_good(phrase):
                    # 参考《文本挖掘从小白到精通（二十四）---如何基于上下文语境提取关键词/关键短语》
                    # 使用sentence-transformers包，将文档和候选词汇及候选短语转化为语义向量，用于后续语义相似度计算
                    candidate_embeddings = request_sbert(
                        "http://10.231.135.106:8096/sbert", phrase
                    )
                    distances = cosine_similarity(
                        doc_embedding, candidate_embeddings
                    )
                    if distances[0][0] > 0.9:
                        kw_dict["词"].append(phrase)
                        kw_dict["权重"].extend(
                            list(map(lambda x: round(x, 3), distances[0]))
                        )

        # 若上述条件未成功抽取，取第一个出现的名词
        if tmp_len == len(kw_dict["词"]):
            im_tree = lac.run(text)  # LAC词语重要性分析 # [[[]]]
            # print(*im_tree)
            for tupl in zip(*im_tree):
                w, pos, im = tupl
                if (
                    pos in ["n", "nz", "nw", "ORG"]
                    and is_good(w)
                    and w + "/n" in zhongzi
                    and im == 3
                ):
                    if not w in kw_dict["词"]:
                        candidate_embeddings = request_sbert(
                            "http://10.231.135.106:8096/sbert", w
                        )
                        distances = cosine_similarity(
                            doc_embedding, candidate_embeddings
                        )
                        if distances[0][0] > 0.9:
                            kw_dict["词"].append(w)
                            kw_dict["权重"].extend(
                                list(map(lambda x: round(x, 3), distances[0]))
                            )
                            break

    # print(kw_dict)
    # 写入文件
    df1 = pd.DataFrame(kw_dict)
    df1.to_excel(writer, sheet_name="关键短语", index=False)
    return kw_dict["词"]


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


# ======================================= #


def w2v_train(key_phrases):

    # 装载分词模型
    lac = LAC(mode='seg')
    for w in key_phrases:
        lac.add_word(w)

    class DW:
        def __iter__(self):
            for l in text_list:
                # yield tokenizer.tokenize(l, combine_Aa123=False)
                yield lac.run(l)

    from gensim.models import Word2Vec
    word2vec = Word2Vec(DW(), vector_size=word_size, min_count=2, sg=1, negative=10)
    return word2vec


def get_start_words():
    # 种子词汇获取（利用已有 tag_name）
    start_words = util.zhongzi[:]
    start_words = list(sorted(set(start_words), key=start_words.index))
    start_words = list(set(start_words) & set(word2vec.wv.index_to_key))   # 获得当前的所有词向量表
    return start_words


import numpy as np
from multiprocessing.dummy import Queue

def most_similar(word, topn, center_vec=None, neg_vec=None):
    """根据给定词、中心向量和负向量找最相近的词
    """
    vec = word2vec.wv[word] + center_vec - neg_vec
    return word2vec.wv.similar_by_vector(vec, topn=topn)


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
    
    c2_file = root + flow_code + "_品牌别名抽取结果.txt"
    cf = open(c2_file, mode="w", encoding="utf-8")

    queue_count = 1
    task_count = 0
    cluster = []
    queue = Queue() # 建立队列
    for w in start_words:
        queue.put((0, w))
        if w not in cluster:
            cluster.append(w)
    while not queue.empty():
        length = int(queue.qsize())
        for _ in range(length):
            idx, word = queue.get()
            cf.write(word + "\t")
            queue_count -= 1
            task_count += 1
            sims = most_similar(word, 10, center_vec, neg_vec)
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
                        cf.write(i+",")
            cf.write("\n")
        cf.write("===========================\n")

    cf.close()
    return cluster


if __name__ == "__main__":

    flow_code = ["LZZXKF", "ZHJKF"][0]
    root = "data/"

    # # 测试用
    # root = "data/"
    # text_list = [
    #     "如何抵扣 App store 充值卡",
    #     "如何抵扣并使用芒果TV会员卡？",
    #     "网易云音乐 VIP卡 是否可以开发票？",
    #     "兑付的Keep会员卡如何使用",
    #     "花加鲜花卡如何使用",
    #     "喜茶礼品卡如何使用",
    #     "充值失败什么时候能退回呢？",
    #     "为什么我一直看不到预定页面?",
    #     "装修+软装如何申请售后？",
    #     "一珑珠等于几京豆",
    #     "为什么我在充值的时候，提示我“运营商维护中，请稍后重试”？",
    #     "手机充值后能取消吗？",
    #     "龙湖酒店代金券以外付款的金额是否可以开具发票？"
    # ]
    # phrase_extract_ddparse(text_list, None)


    from data.C2 import util
    flow_code = "C2ZXKF"
    text_list = util.text_list
    # text_list = text_list[:1000]

    print("数据预处理 ing......")
    # 获取停用词
    data = open("data/中文停用词表.txt", "r", encoding="utf-8").read()
    stopword = data.split("\n")
    lac = LAC(mode="seg")
    sen_list = lac.run(text_list)
    text_list = []
    for sen in sen_list:
        text_list.append("".join([w for w in sen if w not in stopword]))
    data, lac, sen_list = None, None, None

    # excel_filepath = root + flow_code + "领域短语挖掘结果.xlsx"
    # writer = pd.ExcelWriter(excel_filepath)

    # # 主函数
    # # phrase_extract_textrank4zh(text_list, writer)
    # phrase_extract_ddparse(text_list, writer)

    # writer.save()
    # print("结果写入完成！")
    # print("++==============================++")

    # 参照苏神的《分享一次专业领域词汇的无监督挖掘》，通过语义筛选进一步获取领域词
    # https://spaces.ac.cn/archives/6540
    excel_filepath = root + flow_code + "领域短语挖掘结果.xlsx"
    res = read_excel(excel_filepath, 0)
    key_phrase = [_[0] for _ in res]
    # w2v 训练
    word_size = 100
    word2vec = w2v_train(key_phrase)
    start_words = get_start_words()
    cluster_words = find_words(start_words, min_sim=0.6, alpha=0.35)
    print(cluster_words)
    
    
