# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : utils.py
#   Last Modified : 2021-11-28 15:16
#   Describe      : 
#
# ====================================================

import numpy as np


def truncate_sequences(maxlen, indices, *sequences):
    """
    截断直至所有的sequences总长度不超过maxlen
    参数:
        maxlen:
            所有sequence长度之和的最大值 
        indices:
            truncate时删除单词在sequence中的位置
        sequences:
            一条或多条文本
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            # 对sequence中最长的那个进行truncate
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences

