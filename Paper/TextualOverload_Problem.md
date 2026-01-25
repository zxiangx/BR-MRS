文本过载问题(Textual Overload Problem) 
Empirical Observations.
在本节中, 我们通过三层分析, 在半合成数据上的相关性分析, 因果干预分析, 在真实数据上的分析, 考察现有的多模态时间序列预测模型, 究竟捕获了哪些信息。 

我们先观察, 
[
\text{Attn}*{\text{signal}}(t)=\sum*{i\in \mathcal{S}*t}\alpha*{t,i},\quad
\text{Attn}*{\text{noise}}(t)=\sum*{i\notin \mathcal{S}*t}\alpha*{t,i}
]
这个很容易被质疑：噪声 token 多，质量自然大。
长度归一化, ：
[
\bar{\alpha}*{\text{signal}}(t)=\frac{1}{|\mathcal{S}*t|}\sum*{i\in \mathcal{S}*t}\alpha*{t,i},\quad
\bar{\alpha}*{\text{noise}}(t)=\frac{1}{|\mathcal{N}*t|}\sum*{i\in \mathcal{N}*t}\alpha*{t,i}
]
然后报告：
[
R^{\text{attn}}(t)=\log \frac{\bar{\alpha}*{\text{signal}}(t)+\epsilon}{\bar{\alpha}*{\text{noise}}(t)+\epsilon}
]
实验的一些小细节

蓝色表示加入文本后性能上升的样本, 橙色表示加入文本后性能下降的样本, 我们发现, 
证据1 : 
我们发现, 预测性能增长的曲线, Rxxx指标总是的(关注到有价值的token了)
我们发现, 预测性能降低的曲线, xxx指标总是xxx的(忽略掉有价值的token了) 
要做的事情 
1. 定义一个指标, 指标必须深入解释为什么是合理的. 
2. 猜测实验现象, 画图 


证据2 :  信息密度下降
我们逐步从价值token的基础上增加更多的token, 合成一个更长的更完整的句子. 
我们发现随着增长, 信息增益在早期出现一个非常非常短的上涨, 后续出现一个非常快速的下跌

证据3:
