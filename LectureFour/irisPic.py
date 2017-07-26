
#-*- coding:utf -8-*-
#author = "Rachel"
from sklearn.datasets import load_iris   #处理数据
import matplotlib.pyplot as plt #绘图
#加载数据集
iris = load_iris()
print iris.keys()
#样本数和特征值维数
n_samples,n_features= iris.data.shape
print "Number of sample:",n_samples
print "Number of feature:",n_samples
#查看数据集
print iris.data[0]
print iris.data.shape
print iris.target.shape
print iris.target
print iris.target_names


#直方图
x_index=3 #以第三个索引为划分依据
color=["r","#000040","#ff5a00"]
for label,color in zip(range(len(iris.target_names)),color):#zip生成列表对
    plt.hist(iris.data[iris.target==label,x_index],label=iris.target_names[label],color=color)
plt.xlabel(iris.feature_names[x_index])
plt.legend(loc= "upper right")
# plt.show()

#散点图，第一维数据做X轴，第二维做y轴
x_index= 0
y_index=1
color=["blue","red","green"]
for label,color in zip(range(len(iris.target_names)),color):
    plt.scatter(iris.data[iris.target==label,x_index],
    iris.data[iris.target==label,y_index],
    label= iris.target_names[label],
    c = color)
plt.xlabel(iris.feature_names[x_index])
plt.xlabel(iris.feature_names[y_index])
plt.legend(loc= "upper left")
plt.show()





