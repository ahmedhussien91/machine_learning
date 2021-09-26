import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Every pixel is a feature,our linear equation will be ->  y = p1*w1 +p2*w2 + ....... + pn*wn +wo
# constants
NUM_OF_IMAGES_PER_CLASS = 240
TOTAL_NUM_OF_IMAGES = 2400
IMAGE_SIZE = 28 * 28
NUM_OF_CLASSES = 10

# read labels
with open("../Train/Training Labels.txt") as fileObj:
    # fileStr = fileObj.read()
    labelList = fileObj.readlines()
    labelList = [str.strip('\n') for str in labelList]

# load the data of the images into arrays
imList = []
for img_index in range(TOTAL_NUM_OF_IMAGES):
    imList.append(np.reshape(cv2.imread("../Train/" + str(img_index + 1) + ".jpg", 0), (1, IMAGE_SIZE)))

# Separate classes
imgClass_dic = {}
for index in range(NUM_OF_CLASSES):
    imgClass_dic.update({index: imList[NUM_OF_IMAGES_PER_CLASS * index:NUM_OF_IMAGES_PER_CLASS * (index + 1)]})

# calculating mean for 10 classes
mean = []
for index in range(NUM_OF_CLASSES):
    mean.append(np.zeros(IMAGE_SIZE))
    for np_array_28x28 in imgClass_dic[index]:
        mean[index] = mean[index] + np_array_28x28
    mean[index] /= NUM_OF_IMAGES_PER_CLASS

# calculating W for each class vs other classes
W = []
Wo = []
for index in range(NUM_OF_CLASSES):
    # mean of class one Single Class
    m1 = np.matrix(mean[index])
    m1_d = m1.reshape((28,28))
    # mean of others class
    m2 = np.matrix(np.zeros((1, IMAGE_SIZE)))
    for idx, class_mean in enumerate(mean):
        if idx != index:
            m2 = m2 + mean[idx]
    m2 = m2 / (NUM_OF_CLASSES - 1)

    # Class variance for first class
    Class1_var = np.zeros((1, IMAGE_SIZE))
    for n in range(NUM_OF_IMAGES_PER_CLASS):
        tmp = (imgClass_dic[index][n] - m1)
        Class1_var = Class1_var + (tmp.T * tmp)

    # Class Variance for the others class
    Class2_var = np.zeros((1, IMAGE_SIZE))
    for idx in range(NUM_OF_CLASSES):
        if idx != index:
            for n in range(NUM_OF_IMAGES_PER_CLASS):
                tmp = imgClass_dic[idx][n] - m2
                Class2_var = Class2_var + (tmp.T * tmp)

    # Calculating the Sw
    Sw = Class1_var + Class2_var

    W.append((m1 - m2) * np.linalg.pinv(Sw))
    Wo.append(np.matrix((m1+m2)/2) * -W[index].T )

# testing
NUM_OF_TESTING_IMAGES = 200

# read labels
with open("../Test/Test Labels.txt") as fileObj:
    # fileStr = fileObj.read()
    labelList = fileObj.readlines()
    labelList = [str.strip('\n') for str in labelList]

# load the data of the images into arrays
imList = []
for img_index in range(NUM_OF_TESTING_IMAGES):
    imList.append([labelList[img_index], np.reshape(cv2.imread("../Test/" + str(img_index + 1) + ".jpg", 0), (1, IMAGE_SIZE))])

# apply Equation to find Y
y = [[]]
y_Decision = []
for idx,image in enumerate(imList):
    y.append([])
    for index in range(NUM_OF_CLASSES):
        y[idx].append(np.matrix(image[1]) * W[index].T + Wo[index])
    y_Decision.append(y[idx].index(max(y[idx])))

# Create Confusion Matrix
y_actual = pd.Series(labelList,name= "Actual")
y_predicted = pd.Series(y_Decision, name="Predicted")
df_confusion = pd.crosstab(y_actual, y_predicted)
print(df_confusion)

# Draw Confusion Matrix as a heat map
for y in range(df_confusion.values.shape[0]):
    for x in range(df_confusion.values.shape[1]):
        plt.text(x, y, '%d' % df_confusion.values[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.imshow(df_confusion.values[:, :], cmap='hot', interpolation='nearest')
plt.show()
plt.savefig("Confusion.jpg")

# calculate Accuracy
Accuracy = sum(df_confusion.values[i, i] for i in range(NUM_OF_CLASSES))/NUM_OF_TESTING_IMAGES
print("Accuracy= ", Accuracy)
