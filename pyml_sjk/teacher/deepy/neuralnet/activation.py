#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:36:16 2018

@author: seongjoo
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(a):
    c = np.max(a)
    ea = np.exp((a-c).astype(np.float))
    return ea / np.sum(ea)

def cross_entropy_error(y, y_pred):
    delta = 1e-7 # np.log(0) --> - inf
    return -np.sum(y * np.log(y_pred+delta))