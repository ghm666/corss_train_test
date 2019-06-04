"""
第一折交叉验证测试
"""


from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

img_rows, img_cols = 224, 224
nb_classes = 3

model = load_model('../model_save/cross1_step2/vgg_step2_15.h5')

def preprocess_input_vgg(x):
    X = preprocess_input(x)
    return X

# 加载数据集
immatrix = np.load("../numpy_data/immatrix.npy")
imlabel = np.load("../numpy_data/imlabel.npy")
print("origen-immatrix: ",immatrix.shape)
print("origen-imlabel: ",imlabel.shape)

# 打乱数据
data,label = shuffle(immatrix,imlabel, random_state=2)
print(label[:30])

# 选取训练集、测试集
x_test = data[2667:, :]
y_test = label[2667:]

# 训练集和测试集
x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows, 3)

print("new_test_data:",x_test.shape)
print("new_test_label:",y_test.shape)

x_test = x_test.astype('float32')
x_test = preprocess_input_vgg(x_test)

# x_train /= 255
# x_test /= 255

# convert class vectors to binary class matrices
y_test = np_utils.to_categorical(y_test, nb_classes)

def plot_roc():
    """
    该函数用于绘制ROC曲线
    :return:
    """

    # 网络测试的过程中无法一次性测试所有的测试数据
    # 分批次测试数据并将结果拼接到一起
    pred1 = model.predict(x_test[0:100, :], batch_size=100, verbose=0)
    pred2 = model.predict(x_test[100:200, :], batch_size=100, verbose=0)
    pred3 = model.predict(x_test[200:295, :], batch_size=95, verbose=0)
    a1 = np.vstack((pred1,pred2))
    a2 = np.vstack((a1,pred3))
    print(a2.shape)

    y_pred = a2
    y_true = y_test

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('BRVO-CRVO-Normal')
    plt.legend(loc="lower right")
    plt.show()

def matrix():
        """
        该函数用于输出混淆矩阵
        :return:
        """

        y_true = y_test
        y_true = np.argmax(y_true, axis=1)

        pred1 = model.predict(x_test[0:100, :], batch_size=100, verbose=0)
        pred2 = model.predict(x_test[100:200, :], batch_size=100, verbose=0)
        pred3 = model.predict(x_test[200:295, :], batch_size=95, verbose=0)
        a1 = np.vstack((pred1, pred2))
        a2 = np.vstack((a1, pred3))
        y_pred = np.argmax(a2, axis=1)


        target_names = ['normal', 'brvo', 'crvo']
        print("***********classification_report*****************")
        print(classification_report(y_true, y_pred, target_names=target_names))

        # 三分类及二分类数据
        cm = confusion_matrix(y_true, y_pred).T
        print("\n**********3class_confusion_matrix*************")
        print("横轴为预测值（normal、brvo、crvo） 纵轴为实际值 ")
        print(cm)
        TN = cm[0][0]
        FN = cm[0][1] + cm[0][2]
        FP  = cm[1][0] + cm[2][0]
        TP  = cm[1][1] + cm[1][2] + cm[2][1]+ cm[2][2]



        print("\n**********2class_confusion_matrix*************")
        print("横轴为预测值（阳性、阴性） 纵轴为实际值 ")
        print(TP,"  ",FP)
        print(FN,"  ",TN)



        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        specificity = TN * 1.0 / (TN + FP)
        f1_score = (2*precision*recall)/(precision+recall)
        print("precision :",precision)
        print("Sensitivity/recall :",recall)
        print("specificity :",specificity)
        print("F1_score:",f1_score)

def test():
    """
    该函数用于测试模型的精确度
    :return:
    """
    score = model.evaluate(x_test, y_test,batch_size=64,verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    # plot_roc()
    # test()
    matrix()