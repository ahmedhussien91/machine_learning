import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Every pixel is a feature,our linear equation will be ->  y = p1*w1 +p2*w2 + ....... + pn*wn +wo
# constants
NUM_OF_IMAGES_PER_CLASS = 240
TOTAL_NUM_OF_IMAGES = 2400
IMAGE_SIZE = 28 * 28
NUM_OF_CLASSES = 10
NUM_OF_TESTING_IMAGES = 200

# load the data of the images into arrays
imList = []
for img_index in range(TOTAL_NUM_OF_IMAGES):
    imList.append(np.reshape(cv2.imread("../Train/N" + str(img_index + 1) + ".jpg", 0), (1, IMAGE_SIZE)))

# Separate classes
imgClass_dic = {}
for index in range(NUM_OF_CLASSES):
    imgClass_dic.update({index: imList[NUM_OF_IMAGES_PER_CLASS * index:NUM_OF_IMAGES_PER_CLASS * (index + 1)]})

# and don't repeat the operation if the file exist
if not os.path.exists("./Distance_mat.txt"):
    # calculate Distance between Images and create Distance matrix for each image and save it in a file
    Distance_Matrix = np.zeros((TOTAL_NUM_OF_IMAGES,TOTAL_NUM_OF_IMAGES))
    Distance_Matrix = Distance_Matrix.astype('float64')
    for prim_index, primary_image in enumerate(imList):
        for other_index, other_images in enumerate(imList):
            tmp = (primary_image.astype(np.int16) - other_images.astype(np.int16))
            Distance_Matrix[prim_index,other_index] = np.linalg.norm(tmp)
    np.savetxt("Distance_mat.txt", Distance_Matrix, delimiter=',')

# read Distance Matrix into Pandas DataFrame
Distance_mat_DF = pd.read_csv("Distance_mat.txt", names= [x for x in range(TOTAL_NUM_OF_IMAGES)])
print(Distance_mat_DF.head())
# sort each point and create sorted array
Distance_mat_DF_Arr = []
for idx in range(TOTAL_NUM_OF_IMAGES):
    print(Distance_mat_DF[idx].sort_values().head())
    Distance_mat_DF_Arr.append(Distance_mat_DF[idx].sort_values())

#Try Different Values of K
K_range = np.arange(100)+1
if not os.path.exists("./K_Error.txt"):
    K_Error = np.zeros(100)
    for K in K_range:
        # for each image see the nearest neighbours and decide if it was classified right
        # Image right classification happen when int(Image_Index/240) count is more than other occurred indices
        for image_idx in range(TOTAL_NUM_OF_IMAGES):
            classes_count = np.zeros(NUM_OF_CLASSES)
            for neighbour_index in range(K):
                classes_count[int(list(Distance_mat_DF_Arr[image_idx].index)[neighbour_index+1] / 240)] += 1
            if not (list(classes_count).index(max(list(classes_count))) == int(list(Distance_mat_DF_Arr[image_idx].index)[0]/240)):
                K_Error[K-1]+=1
        print("Error of K = ",K ," is ",K_Error[K-1])

    # plot K_errors
    np.savetxt("K_Error.txt", K_Error, delimiter=',')
K_Error = np.genfromtxt("K_Error.txt", delimiter=',')
plt.plot(K_range,K_Error)
plt.savefig("KNN.png")
K = list(K_Error).index(K_Error.min())+1
print("K=",K)
# K=1

# load the data of the images into arrays
distance_list = np.zeros(TOTAL_NUM_OF_IMAGES)
y_Decision = []
for img_index in range(NUM_OF_TESTING_IMAGES):
    classes_count = np.zeros(NUM_OF_CLASSES)
    testing_image = np.reshape(cv2.imread("../Test/N" + str(img_index + 1) + ".jpg", 0), (1, IMAGE_SIZE))
    for other_index, other_images in enumerate(imList):
        tmp = (testing_image.astype(np.int16) - other_images.astype(np.int16))
        distance_list[other_index] = np.linalg.norm(tmp)
    testing_distance_series = pd.Series(distance_list).sort_values()
    for neighbour_index in range(K):
        classes_count[int(list(testing_distance_series.index)[neighbour_index+1]/240)] +=1
    y_Decision.append(list(classes_count).index(max(list(classes_count))))

# read labels
with open("../Test/Test Labels.txt") as fileObj:
    labelList = fileObj.readlines()
    labelList = [str.strip('\n') for str in labelList]

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
plt.savefig("Confusion.png")

# calculate Accuracy
Accuracy = sum(df_confusion.values[i, i] for i in range(NUM_OF_CLASSES))/NUM_OF_TESTING_IMAGES
print("Accuracy= ", Accuracy)