"""
交叉验证第二折-----vgg16第二阶段训练
"""

from sklearn.utils import shuffle
import numpy as np

from keras.utils import np_utils
from keras.models import load_model
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications.vgg16 import preprocess_input

# 参数设置
img_rows, img_cols = 224, 224
nb_classes = 3
batchsize= 64
train_epoch= 16

# 加载数据集
immatrix = np.load("../numpy_data/immatrix.npy")
imlabel = np.load("../numpy_data/imlabel.npy")
print("origen-immatrix: ",immatrix.shape)
print("origen-imlabel: ",imlabel.shape)


# 显示训练图像
# import matplotlib.pyplot as plt
# for i in range (10):
#     img=immatrix[i].reshape(img_rows,img_cols,3)
#     print('severity',imlabel[i])
#     plt.imshow(img)
#     plt.show()

# 打乱数据
data,label = shuffle(immatrix,imlabel, random_state=2)
print(label[:30])

# 选取训练集、测试集
x_train = data[295:, :]
x_test = data[:295, :]
y_train = label[295:]
y_test = label[:295]




# 训练集和测试集
x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, 3)
x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows, 3)

print("new_train_data:",x_train.shape)
print("new_train_label:",y_train.shape)
print("new_test_data:",x_test.shape)
print("new_test_label:",y_test.shape)
#
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# x_train /= 255
# x_test /= 255

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# import matplotlib.pyplot as plt
# for i in range (10):
#     img=x_train[i].reshape(img_rows,img_cols,3)
#     print('severity',y_train[i])
#     plt.imshow(img)
#     plt.show()

def preprocess_input_vgg(x):
    X = preprocess_input(x)
    return X

vgg_model = load_model('../model_save/cross2_step1/vgg_step1_20.h5')
vgg_model.summary()


for i, layer in enumerate(vgg_model.layers):
    print(i, layer.name)


for layer in vgg_model.layers[:16]:
    layer.trainable = False
for layer in vgg_model.layers[16:]:
    layer.trainable = True

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg,
                                   rotation_range=90,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   vertical_flip=True,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow(x_train,y_train,batch_size=batchsize)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
validation_generator = validation_datagen.flow(x_test,y_test,batch_size=batchsize)


sgd = SGD(lr=1e-3, momentum=0.9)
vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


filepath="../model_save/vgg_step2_{epoch:02d}.h5"
checkpointer1 = keras.callbacks.ModelCheckpoint(filepath,
                               monitor='loss',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=False,
                               mode='auto',
                               period=2)


tensroboad = keras.callbacks.TensorBoard(log_dir='../logs')


vgg_model.fit_generator(train_generator, steps_per_epoch=int(len(x_train)/batchsize),
                    epochs=train_epoch, validation_data=validation_generator,
                    validation_steps=int(len(x_test)/batchsize),
                    callbacks = [checkpointer1, tensroboad])


