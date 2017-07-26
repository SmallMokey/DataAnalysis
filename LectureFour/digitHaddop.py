#-*-coding:utf-8-*-
#author = "Rachel"
from sklearn.datasets import load_digits  #加载数据集
from sklearn import svm  #选择svm模型
from sklearn.metrics import accuracy_score
import pickle #模型加载和保存
import numpy as np #生成数据集
import  pandas as pd
#加载数据集
digits = load_digits()
#查看数据集
print digits.data
print digits.images.shape
data = digits.images.reshape((digits.images.shape[0],-1))
# digits.data = pd.DataFrame(digits.data)
print digits.data.shape
print digits.target_names
print digits.target
#在训练集上训练模型
svm_classifier = svm.SVC(gamma=1,C = 100)#Estimator对象
# svm_classifier = svm.SVC(gamma=0.1,C = 1000)
#手动划分训练集和测试集
n_test = 100 #测试样本个数
train_x = digits.data[:-n_test,:]
train_y = digits.target[:-n_test]

test_x = digits.data[-n_test:,:]
y_true = digits.target[-n_test:]


#训练模型
svm_classifier.fit(train_x,train_y)

#在测试集上测试模型
y_pred = svm_classifier.predict(test_x)
print accuracy_score(y_true,y_pred)

#保存模型
with open("D:/testResult/model/svm_model.pkl","wb") as f:
    pickle.dump(svm_classifier,f)

#重新加载模型进行预测
with open("D:/testResult/model/svm_model.pkl","rb") as f:
    model = pickle.load(f)
random_sample_index=np.random.randint(0,1796,10)
random_samples = digits.data[random_sample_index,:]
random_target = digits.target[random_sample_index]

random_predict= model.predict(random_samples)
print random_target
print random_predict
print accuracy_score(random_predict,random_target)
#绘图
import matplotlib.pyplot as plt
# color= ["#123456","r","m","y","g","k","b","#000040","#FFa500","#ff0000"]
# for image,color in zip((range(len(train_x)),color)):
#     plt.hist(train_x,train_y)
a = range(10)
plt.plot(a,random_target,label = "True")
plt.plot(a,random_predict,label="prediction")
# plt.fill_between(a,random_target,random_predict,where= (random_predict > random_target),facecolor='y',edgecolor="k",alpha=0.5)
# plt.fill_between(a,random_target,random_predict,where= (random_predict < random_target),facecolor="#ffa500",edgecolor="k",alpha=0.5)
# plt.fill_between(a,random_target,random_predict,where= (random_predict == random_target),facecolor="c",edgecolor="k",alpha=0.5)
plt.xlabel("imageID")
plt.ylabel("DigitsClass")
plt.title("accuracy")
plt.show()











