# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:57:23 2019

@author: Mechrevo
"""

from sklearn import svm
import numpy as np
from sklearn import model_selection

def chag_type(z):
    chag = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
    return chag[z] 

def iris_type(s):
    class_label={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}
    return class_label[s]

filepath='C:\\Users\\Mechrevo\\HelloWord\\Iris\\iris.data'  # 数据文件路径
data=np.loadtxt(filepath,dtype=float,delimiter=',',converters={4:iris_type})

X ,y=np.split(data,(4,),axis=1) #np.split 按照列（axis=1）进行分割，从第四列开始往后的作为y 数据，之前的作为X 数据。函数 split(数据，分割位置，轴=1（水平分割） or 0（垂直分割）)。
x=X[:,:4] #在 X中取前两列作为特征（为了后期的可视化画图更加直观，故只取前两列特征值向量进行训练）
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=1,test_size=0.6)
classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovr',C=0.8)
classifier.fit(x_train,y_train.ravel())

print("SVM-输出训练集的准确率为：",classifier.score(x_train,y_train))

print("SVM-输出测试集的准确率为：",classifier.score(x_test,y_test))

print('\npredict:\n', classifier.predict(x_train))

while(1):
    print("\n请输入花的四个特征参数,以空格分隔：")
    feature = [[float(temp) for temp in input().split(' ')]]
    type_num=classifier.predict(feature)
    typ=chag_type(type_num[0])
    print("花的种类为",typ)