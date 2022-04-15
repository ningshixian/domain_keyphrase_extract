import re
import codecs
from tqdm import tqdm
import csv
import pandas as pd

from autophrasex import *
from autophrasex.tokenizer import AbstractTokenizer
import pandas as pd
import tqdm
from LAC import LAC
import jionlp as jio

"""
pip install -U autophrasex
https://github.com/luozhouyang/AutoPhraseX
"""


# 第一步：种子词汇获取（利用已有 tag_name）
zhongzi = []
with codecs.open("data/LZZXKF_tag.csv", 'r', encoding="utf-8") as f:
    reader = csv.reader(f)
    reader.__next__()
    for row in reader:
        zhongzi.append(row[0])
        for sim in row[1].split("###"):
            if sim and len(sim)>1 and not sim.encode("utf-8").isalnum():
                zhongzi.append(sim)
zhongzi = list(sorted(set(zhongzi), key=zhongzi.index))
with codecs.open('data/LZZXKF_tag.txt', 'w', encoding='utf-8') as writer:
    for item in zhongzi:
        writer.write(item + "\n")


def read_excel(f_path, sheet):
    """读取Excel文件"""
    exc = pd.io.excel.ExcelFile(f_path)
    df = pd.read_excel(exc, sheet_name=sheet, dtype=str)
    df.fillna("", inplace=True)
    df_li = df.values.tolist()
    return df_li


# 待挖掘文本处理成特定格式
res = read_excel("data/LZZXKF知识.xlsx", 0)
text_list = []
for item in res:
    pri, sim = item[0], item[1]
    if pri:
        text_list.append(pri)
    if sim:
        text_list.extend(sim.split("###"))
print(len(text_list))
text_list = text_list[:]
with codecs.open("data/knowledge.txt", "w", encoding="utf-8") as writer:
    for line in text_list:
        writer.write(line)
        writer.write("\n")


# 通过jionlp的关键词提取，得到自定义词典（避免被分词）
words = ["马上修/n", "喜茶/n"]  
others = []
for text in text_list:
    keyphrases = jio.keyphrase.extract_keyphrase(text, top_k=1)
    # tr4w.analyze(text, window=2)  # , lower=True
    # keyphrases = tr4w.get_keyphrases(keywords_num=20, min_occur_num=1)
    others.append(list(map(lambda x: x+"/n", keyphrases)))
words = list(set(words + [y for x in others for y in x]))


# 自定义分词器
class BaiduLacTokenizer(AbstractTokenizer):

    def __init__(self, custom_vocab_path=None, model_path=None, mode='seg', use_cuda=False, **kwargs):
        self.lac = LAC(model_path=model_path, mode=mode, use_cuda=use_cuda)
        for w in words:
            self.lac.add_word(w)
        if custom_vocab_path:
            self.lac.load_customization(custom_vocab_path)

    def tokenize(self, text, **kwargs):
        text = self._uniform_text(text, **kwargs)
        results = self.lac.run(text)
        results = [x.strip() for x in results if x.strip()]
        return results


# 自定义过滤器
class MyPhraseFilter(AbstractPhraseFilter):

    def apply(self, pair, **kwargs):
        phrase, freq = pair
        # return True to filter this phrase
        if is_verb(phrase):
            return True
        return False


# 构造autophrase
# tokenizer用于文本分词，这里使用baidu/LAC来进行中文分词
# reader用于读取语料
# selector可以拥有多个phrase_filter，用于实现Phrase的过滤
# extractor用于抽取分类器的特征
autophrase = AutoPhrase(
    # reader=DefaultCorpusReader(tokenizer=JiebaTokenizer()),
    reader=DefaultCorpusReader(tokenizer=BaiduLacTokenizer()),
    selector=DefaultPhraseSelector(min_freq=1),
    extractors=[
        NgramsExtractor(N=4),   # n-gram特征抽取器，可以计算phrase的pmi特征
        IDFExtractor(),         # 计算phrase的doc_freq、idf特征
        EntropyExtractor()      # 计算phrase的左右熵特征
    ]
)

# 开始挖掘
predictions = autophrase.mine(
    corpus_files=['./data/knowledge.txt'],
    quality_phrase_files='./data/LZZXKF_tag.txt',
    callbacks=[
        LoggingCallback(),                      # 提供挖掘过程的日志信息打印
        ConstantThresholdScheduler(),           # 在训练过程中调整阈值的回调
        EarlyStopping(patience=2, min_delta=3)  # 早停，在指标没有改善的情况下停止训练
    ])

# 输出挖掘结果
words = []
prob = []
for pred in tqdm.tqdm(predictions):
    words.append(pred[0])
    prob.append(pred[-1])
df = pd.DataFrame({"words":words, "prob":prob})
df.to_csv('./data/mining_phrase.csv', index=None)
print('ok')
