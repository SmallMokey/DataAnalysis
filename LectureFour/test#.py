#-*-coding:utf-8-*-
#author = "Rachel"
#生成多类单标签数据集
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
"""
生成数据集：可以用来分类任务，可以用来回归任务，可以用来聚类任务，用于流形学习的，用于因子分解任务的
用于分类任务和聚类任务的：这些函数产生样本特征向量矩阵以及对应的类别标签集合
make_blobs：多类单标签数据集，为每个类分配一个或多个正太分布的点集
make_classification：多类单标签数据集，为每个类分配一个或多个正太分布的点集，提供了为数据添加噪声的方式，包括维度相关性，无效特征以及冗余特征等
make_gaussian-quantiles：将一个单高斯分布的点集划分为两个数量均等的点集，作为两类
make_hastie-10-2：产生一个相似的二元分类数据集，有10个维度
make_circle和make_moom产生二维二元分类数据集来测试某些算法的性能，可以为数据集添加噪声，可以为二元分类器产生一些球形判决界面的数据
"""
center=[[1,1],[-1,-1],[1,-1]]
cluster_std=0.3
X,labels=make_blobs(n_samples=200,centers=center,n_features=2,
                    cluster_std=cluster_std,random_state=0)
print('X.shape',X.shape)
print("labels",set(labels))

unique_lables=set(labels)
colors=plt.cm.Spectral(np.linspace(0,1,len(unique_lables)))
for k,col in zip(unique_lables,colors):
    x_k=X[labels==k]
    plt.plot(x_k[:,0],x_k[:,1],'o',markerfacecolor=col,markeredgecolor="k",
             markersize=14)
plt.title('data by make_blob()')
plt.show()

#生成用于分类的数据集
from sklearn.datasets.samples_generator import make_classification
X,labels=make_classification(n_samples=200,n_features=2,n_redundant=0,n_informative=2,
                             random_state=1,n_clusters_per_class=2)
rng=np.random.RandomState(2)
X+=2*rng.uniform(size=X.shape)

unique_lables=set(labels)
colors=plt.cm.Spectral(np.linspace(0,1,len(unique_lables)))
for k,col in zip(unique_lables,colors):
    x_k=X[labels==k]
    plt.plot(x_k[:,0],x_k[:,1],'o',markerfacecolor=col,markeredgecolor="k",
             markersize=14)
plt.title('data by make_classification()')
plt.show()

#生成球形判决界面的数据
from sklearn.datasets.samples_generator import make_circles
X,labels=make_circles(n_samples=200,noise=0.2,factor=0.2,random_state=1)
print("X.shape:",X.shape)
print("labels:",set(labels))

unique_lables=set(labels)
colors=plt.cm.Spectral(np.linspace(0,1,len(unique_lables)))
for k,col in zip(unique_lables,colors):
    x_k=X[labels==k]
    plt.plot(x_k[:,0],x_k[:,1],'o',markerfacecolor=col,markeredgecolor="k",
             markersize=14)
plt.title('data by make_moons()')
plt.show()



