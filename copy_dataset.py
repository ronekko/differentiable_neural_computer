# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 00:23:14 2017

@author: sakurai
"""

import numpy as np


def generate_copy_data(batch_size=10, seq_len=10, dim_x=9):
    x = np.zeros((batch_size, seq_len + 1, dim_x), 'f')
    b = np.random.choice((0, 1), size=(batch_size, seq_len, dim_x - 1))
    x[:, :-1, :-1] = b.astype('f')
    x[:, -1, -1] = 1
    t = x.copy().astype('i')
    return x, t


if __name__ == '__main__':
    batch_size = 10
    seq_len = 10
    dim_x = 9

    x, t = generate_copy_data(batch_size, seq_len, dim_x)
