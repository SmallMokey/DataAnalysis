#-*- coding:utf-8 -*-
#author = "Rachel"
import numpy as np
from sklearn.model_selection import train_test_split #划分训练集和测试集
from sklearn.preprocessing import scale  #数据预处理包-特征规一化
from sklearn.datasets import make_classification  #生成分类数据进行验证scale的必要性
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn import datasets
from sklearn.linear_model import LinearRegression
x = np.random.randint(0,100,(10,4))
y = np.random.randint(0,3,10)
y.sort()
print  "samples：",x
print "label:",y

#分割训练集，测试集
#random_state确保每次分割的结果相同,test_size是训练集和测试集之比
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=7)
print "TrainingSet:"
print x_train
print y_train

print "TestingSet:"
print x_test
print y_test

#特征归一化
x1 = np.random.randint(0,1000,(5,1))
x2 = np.random.randint(0,10,(5,1))
x3 = np.random.randint(0,100000,(5,1))
X = np.concatenate([x1,x2,x3],axis = 1)#连接
print X
print scale(X)

#生成分类数据进行验证scale的必要性
X,y = make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,
                          random_state=25,n_clusters_per_class=1,scale=100)
plt.scatter(X[:,0],X[:,1],c = y)
plt.show()

X = preprocessing.scale(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=7)
svm_classifier = svm.SVC()
svm_classifier.fit(X_train,y_train)
svm_classifier.score(X_test,y_test)

#训练模型
boston_data = datasets.load_boston()
X = boston_data.data
y = boston_data.target

print('样本：')
print(X[:5, :])
print('标签：')
print(y[:5])

# 选择线性回顾模型
lr_model = LinearRegression()
# 分割训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3., random_state=7)
# 训练模型
lr_model.fit(X_train, y_train)
# 返回参数
lr_model.get_params()
lr_model.score(X_train, y_train)
lr_model.score(X_test, y_test)

#交叉验证
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3., random_state=10)

k_range = range(1, 31)
cv_scores = []
for n in k_range:
    knn = KNeighborsClassifier(n)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')  # 分类问题使用
    # scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='neg_mean_squared_error') # 回归问题使用
    cv_scores.append(scores.mean())

plt.plot(k_range, cv_scores)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
# 选择最优的K
best_knn = KNeighborsClassifier(30)
best_knn.fit(X_train, y_train)
print(best_knn.score(X_test, y_test))
print(best_knn.predict(X_test))









