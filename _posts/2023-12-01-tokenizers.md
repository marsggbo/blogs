---
title: LLM 入门笔记-Tokenizer
category: 小书匠/帮助手册
renderNumberedHeading: true
slug:  storywriter/tutorial
tags: LLM,大模型,tokenizer,transformers
---

> 以下笔记参考huggingface 官方 tutorial： https://huggingface.co/learn/nlp-course/chapter6



下图展示了完整的 tokenization 流程，接下来会对每个步骤做进一步的介绍。

![tokenizer_pipeline](https://raw.githubusercontent.com/marsggbo/PicBed/master/小书匠/2023_12_1_1701411316087.png)

# 1. Normalization

normalize 其实就是根据不同的需要对文本数据做一下清洗工作，以英文文本为例可以包括删除不必要的空白、小写和/或删除重音符号。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

>>> 'hello how are u?'
```



# 2. Pre-tokenization

数据清洗好后，我们需要将文本作划分。对于英语而言，最简单的划分逻辑就是以单词为单位进行划分。不过即使是这么简单的规则也可以细分出很多不同的划分方式，下面展示了 3 种划分方式，它们适用于不同的模型训练，返回的是一个 list，每个元素是一个tuple。tuple 内第一个元素是划分后的sub-word，第二个元素是其初始和结尾的索引。

- bert 的最简单，真的就是最符合直觉的 huafenfangshi
- gpt2划分的不同点是单词前如果有空格的话，空格会转换成一个特殊字符，即 **Ġ**。
- t5 类似 gpt2 也考虑了空格，不过空格被替换成了 **_**

![normalize](https://raw.githubusercontent.com/marsggbo/PicBed/master/小书匠/2023_12_1_1701411960722.png)


# 3. BPE Tokenization

上面Pre-tokenization展示的是比较简单的划分方式，但是他们的缺点是会导致词表非常大。而且，我们知道英文单词是有词根的，并且一个动词会有不同的时态，简单的以单词为单位划分，不太便于表示单词之间的相似性。所以一种可行的办法是我们寻找单词间的公约数，即把单词拆分成若干个 sub-word。为方便理解，我们可以以 like, liked, liking 为例，这三者的公约数是 lik, 所以分别可以拆分成如下（实际上的拆分并不一定就是下面的结果，这里只是为了方便解释说明）：
- ["lik", "e"]
- ["lik", "ed"]
- ["lik", "ing"]

模型在计算这三个单词的相似性的时候，因为他们具有相同的"lik"，所以肯定会认为有很高的相似性。类似的，当模型计算两个都带有"ed"的单词的时候，也会知道这两个单词也会有相似性，因为都表示过去式。

那么如何寻找公约数呢？大佬们提出了不同的算法，常见的三个算法总结在下表里了：


![BPE-wordpiece-unigram](https://raw.githubusercontent.com/marsggbo/PicBed/master/小书匠/2023_12_1_1701412605182.png)


这一小节我们着重介绍一下最常见的算法之一：BPE (Byte-pair Encoding)。[huggingface官方tutorial](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt) 给出了非常详细的解释，这里做一个简单的介绍。

BPE 其实是一个统计算法，不同意深度神经网络，只要给定一个数据集或者一篇文章，BPE 不管运行多少次都会得出同样的结果。下面我们看看 BPE 到底是在做什么。

为了方便理解，我们假设我们的语料库中只有下面 5 个单词，数字表示出现的频率：

`语料库：[("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)]`

BPE 根据上述单词表首先初始化生成基础词汇表（base vocabulary），即 

`词汇表：["b", "g", "h", "n", "p", "s", "u"]`

我们可以将每个单词看成是一个由多个基础 token 组成的 list，即

`[("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)]`

接下来，正如 BPE 的名字 byte-pair 所表示的，它会对每个单词内部相邻的 token 逐一进行两两匹配，然后找出出现频率最高的那一对，例如，`("h" "u" "g", 10)` 匹配结果会到的 `("h", "u", 10)` 和 `("u", "g", 10)`，其他单词同理。

通过遍历所有单词我们可以发现出现频率最高的 `("u", "g")`，它在 "hug"、"pug" 和 "hugs" 中出现，总共出现了 20 次，所以 BPE 会将它们进行合并（merge），即 `("u", "g") -> "ug"`。这样基础词汇表就可以新增一个 token 了，更新后的词汇表和语料库如下：

```python
词汇表：["b", "g", "h", "n", "p", "s", "u", "ug"]
语料库：("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

我们继续重复上面的 遍历和合并 操作，每次词汇表都会新增一个 token。当词汇表内 token 数量达到预设值的时候就会停止 BPE 算法了，并返回最终的词汇表和语料库。



<footer style="color:white;;background-color:rgb(24,24,24);padding:10px;border-radius:10px;">
<h3 style="text-align:center;color:tomato;font-size:16px;" id="autoid-2-0-0">
<center>
<span>微信公众号：AutoML机器学习</span><br>
<img src="https://pic4.zhimg.com/80/v2-87083e55cd41dbef83cc840c142df48a_720w.jpeg" style="width:200px;height:200px">
</center>
<b>MARSGGBO</b><b style="color:white;"><span style="font-size:25px;">♥</span>原创</b><br>
<span>如有意合作或学术讨论欢迎私戳联系~<br>邮箱:marsggbo@foxmail.com</span>
<b style="color:white;"><br>
</b><p><b style="color:white;"></b>
</p></h3>
</footer>
