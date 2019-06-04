该程序用于brvo、crvo、normal图像的三分类，此外可以进行多折交叉验证。数据的加载方式为使用numpy矩阵加载数据及标签。

数据集--------------train data:2662张，test data:295张

logs------tensorboard文件存放处，用于显示train acc、loss，val acc、loss

model_save---------保存训练各个阶段网络的模型文件

numpy_data--------用于存放制作好的数据集，immatrix.npy：存放图片 imlabel.npy：存放图像对应的标签

result----------存放训练及测试过程中的一些结果

train_test---------存放加载数据、网络训练、网络测试的代码


