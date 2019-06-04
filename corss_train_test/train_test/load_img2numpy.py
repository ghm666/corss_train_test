"""
所需文件train.txt:
                 0/normal1.jpg 0
                 1/brvo1104.jpg 1
                 2/crvo395.jpg 2
txt为标签文件存放图片的路径及标签

将三类别的图像及对应的标签分别存储于numpy矩阵中

"""
from PIL import Image
import numpy as np
import os

txtfile = "F://doctor_labeled_3class//cross_data//train.txt"
img_dir = "F://doctor_labeled_3class//cross_data//"

# 两个列表分别用来存储图像及标签
immatrix = []
imlabel = []

fr = open(txtfile)
for line in fr.readlines():
    print(line)
    line = line.strip()
    listFromLine = line.split(' ')
    imlabel.append(int(listFromLine[1]))


    img_path = os.path.join(img_dir, listFromLine[0])
    print(img_path)
    im = Image.open(img_path)
    # img = im.resize((img_rows,img_cols))
    rgb = im.convert('RGB')
    immatrix.append(np.array(rgb).flatten())
fr.close()

#converting images & labels to numpy arrays
immatrix = np.asarray(immatrix)
imlabel = np.asarray(imlabel)
print(imlabel)
np.save("../numpy_data/immatrix.npy",immatrix)
np.save("../numpy_data/imlabel.npy",imlabel)







