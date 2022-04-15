# -*- coding:utf-8 -*-
"""
@author: ningshixian
@file: plot_util.py.py
@time: 2019/7/25 11:58
"""

# 使用图分析可以更加直观地展示数据的分布（频数分析）和关系（关系分析）
# 柱状图和饼状图是对定性数据进行频数分析的常用工具，使用前需将每一类的频数计算出来
# 直方图和累积曲线是对定量数据进行频数分析的常用工具，直方图对应密度函数而累积曲线对应分布函数
# 散点图可用来对两组数据的关系进行描述
# 在没有分析目标时，需要对数据进行探索性分析，箱形图可以完成这一任务
from gen_data import Gen_data
from matplotlib import pyplot


# 绘制柱状图
def drawBar(grades):
    xticks = ['A', 'B', 'C', 'D', 'E']
    gradeGroup = {}
    # 对每一类成绩进行频数统计
    for grade in grades:
        gradeGroup[grade] = gradeGroup.get(grade, 0) + 1
    print(gradeGroup)
    # 创建柱状图
    # 第一个参数为柱的横坐标
    # 第二个参数为柱的高度
    # 参数align为柱的对齐方式，以第一个参数为参考标准
    pyplot.bar(range(5), [gradeGroup.get(xtick, 0) for xtick in xticks], align='center')

    # 设置柱的文字说明
    # 第一个参数为文字说明的横坐标
    # 第二个参数为文字说明的内容
    pyplot.xticks(range(5), xticks)

    # 设置横坐标的文字说明
    pyplot.xlabel('Grade')
    # 设置纵坐标的文字说明
    pyplot.ylabel('Frequency')
    # 设置标题
    pyplot.title('Grades Of Male Students')
    # 绘图
    pyplot.show()


# 绘制饼形图
def drawPie(grades):
    labels = ['A', 'B', 'C', 'D', 'E']
    gradeGroup = {}
    for grade in grades:
        gradeGroup[grade] = gradeGroup.get(grade, 0) + 1
    print(gradeGroup)
    # 创建饼形图
    # 第一个参数为扇形的面积
    # labels参数为扇形的说明文字
    # autopct参数为扇形占比的显示格式
    pyplot.pie([gradeGroup.get(label, 0) for label in labels], labels=labels, autopct='%1.1f%%')
    pyplot.title('Grades Of Male Students')
    pyplot.show()


# 绘制直方图
def drawHist(heights):
    # 创建直方图
    # 第一个参数为待绘制的定量数据，不同于定性数据，这里并没有事先进行频数统计
    # 第二个参数为划分的区间个数
    pyplot.hist(heights, 100)
    pyplot.xlabel('Heights')
    pyplot.ylabel('Frequency')
    pyplot.title('Heights Of Male Students')
    pyplot.show()


# 绘制累积曲线
def drawCumulativeHist(heights):
    # 创建累积曲线
    # 第一个参数为待绘制的定量数据
    # 第二个参数为划分的区间个数
    # normed参数为是否无量纲化
    # histtype参数为'step'，绘制阶梯状的曲线
    # cumulative参数为是否累积
    pyplot.hist(heights, 20, normed=True, histtype='step', cumulative=True)
    pyplot.xlabel('Heights')
    pyplot.ylabel('Frequency')
    pyplot.title('Heights Of Male Students')
    pyplot.show()


# 绘制散点图
def drawScatter(heights, weights):
    # 创建散点图
    # 第一个参数为点的横坐标
    # 第二个参数为点的纵坐标
    pyplot.scatter(heights, weights)
    pyplot.xlabel('Heights')
    pyplot.ylabel('Weights')
    pyplot.title('Heights & Weights Of Male Students')
    pyplot.show()


# 绘制箱形图
def drawBox(heights):
    # 创建箱形图
    # 第一个参数为待绘制的定量数据
    # 第二个参数为数据的文字说明
    pyplot.boxplot([heights], labels=['Heights'])
    pyplot.title('Heights Of Male Students')
    pyplot.show()


# 绘制箱形图
def drawBox(heights):
    # 创建箱形图
    # 第一个参数为待绘制的定量数据
    # 第二个参数为数据的文字说明
    pyplot.boxplot([heights], labels=['Heights'])
    pyplot.title('Heights Of Male Students')
    pyplot.show()


if __name__ == '__main__':
    gen_data = Gen_data.genData(Gen_data)
    # 频数分析——性分析——柱状图
    # drawBar(gen_data.grades)
    # 频数分析——定性分析——饼状图
    # drawPie(gen_data.grades)
    # 频数分析——定量分析——直方图
    # drawHist(gen_data.heights)
    # 频数分析——定量分析——累积曲线
    # drawCumulativeHist(gen_data.heights)
    # 关系分析——散点图
    # drawScatter(gen_data.heights, gen_data.weights)
    # 探索分析——箱形图
    drawBox(gen_data.heights)
