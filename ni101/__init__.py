"""NI 43-101 Indicated / Inferred 资源量抽取与对抗审核。

本包实现题目 #3 要求的四个交付组件：

* ``ni101.extractor``  —— Extractor Agent（强模型抽取）
* ``ni101.critic``     —— CriticMaster Agent（第二模型挑刺打分 1-10）
* ``ni101.pipeline``   —— Revise Loop（<= 3 轮，评分 >= 8 通过，否则 abstain）
* ``ni101.evolution``  —— Evolution Log（evolution.jsonl 自动落盘 + few-shot 回放）
"""

__version__ = "0.1.0"
