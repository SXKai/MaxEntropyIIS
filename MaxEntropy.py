# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:56:44 2017

@author: Q
"""

import collections
import math

class MaxEntropy():
    def __init__(self):
        self._samples = []   #样本集，元素是[y,x1,x2,...]的样本
        self._Y = set([])  #标签集合，相当去去重后的y
        self._numXY = collections.defaultdict(int)   #key为(x,y)，value为出现次数
        self._N = 0  #样本数
        self._ep_ = []   #样本分布的特征期望值
        self._xyID = {}   #key记录(x,y),value记录id号
        self._n = 0  #特征的个数
        self._C = 0   #最大特征数
        self._IDxy = {}    #key为(x,y)，value为对应的id号  
        self._w = []
        self._EPS = 0.005   #收敛条件
        self._lastw = []    #上一次w参数值
    def loadData(self,filename):
        with open(filename) as fp:
            self._samples = [item.strip().split('\t') for item in fp.readlines()]
        for items in self._samples:
                y = items[0]
                X = items[1:]
                self._Y.add(y)
                for x in X:
                    self._numXY[(x,y)] += 1
    def _sample_ep(self):   #计算特征函数fi关于经验分布的期望
        self._ep_ = [0] * self._n
        for i,xy in enumerate(self._numXY):
            self._ep_[i] = self._numXY[xy]/self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

            
        
    def _initparams(self):  #初始化参数
        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._C = max([len(sample)-1 for sample in self._samples])
        self._w = [0]*self._n
        self._lastw = self._w[:]
        
        self._sample_ep()                 #计算每个特征关于经验分布的期望
    def _Zx(self,X):    #计算每个x的Z值
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x,y) in self._numXY:
                    ss += self._w[self._xyID[(x,y)]]
            zx += math.exp(ss)
        return zx
    
    def _model_pyx(self,y,X):   #计算每个P(y|x)
        Z = self._Zx(X)
        ss = 0
        for x in X:
            if (x,y) in self._numXY:
                ss += self._w[self._xyID[(x,y)]]
        pyx = math.exp(ss)/Z
        return pyx
       
    def _model_ep(self,index):   #计算特征函数fi关于模型的期望
        x,y = self._IDxy[index]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self._model_pyx(y,sample)
            ep += pyx/self._N
        return ep
            
    def _convergence(self):
        for last,now in zip(self._lastw,self._w):
            if abs(last - now) >=self._EPS:
                return False
        return True
    
    def predict(self,X):   #计算预测概率
        Z = self._Zx(X)
        result = {}
        for y in self._Y:
            ss = 0
            for x in X:
                if (x,y) in self._numXY:
                    ss += self._w[self._xyID[(x,y)]]
            pyx = math.exp(ss)/Z
            result[y] = pyx
        return result
         
    def train(self,maxiter = 1000):   #训练数据
        self._initparams()
        for loop in range(0,maxiter):  #最大训练次数
            print ("iter:%d"%loop)
            self._lastw = self._w[:]
            for i in range(self._n):
                ep = self._model_ep(i)    #计算第i个特征的模型期望
                self._w[i] += math.log(self._ep_[i]/ep)/self._C   #更新参数
            print("w:",self._w)
            if self._convergence():   #判断是否收敛
                break
                
                
    
maxent = MaxEntropy()
x = ['sunny','hot','high','FALSE']
maxent.loadData('dataset.txt')
maxent.train()
print('predict::::::::::::::::::',maxent.predict(x))