# PROBLEM
You are required to design a Fisher’s linear discriminant classifier that can recognize scanned 
images of the 10 digits (0 to 9) provided in the file “Assignment 1 Dataset.zip”. The zip file 
contains two folders: “Train” and “Test”. The “Train” folder contains 240 images for each 
digit while the “Test” folder contains 20 images for each digit. The images in the “Train” 
folder should be used to train a classifier for each digit using the method given at the bottom 
of slide 9 in Lecture 2.pdf. The folder contains a filed named “Training Labels.txt” which 
includes the labels of the 2400 images in order. After the classifiers are trained, test each 
classifier using the images given in the “Test” folder. Use the following equation for Fisher’s 
Linear Discriminant w = S𝑊^−1(m2 − m1). The folder also contains a text file named “Test 
Labels.txt” which include the labels of the 200 images in order. 
Deliverables: 
• Your code. 
• A confusion matrix showing the number of images of the Test folder of each digit 
that were classified to belong to different digits (For example: Number of images of 0 
that were classified as 0, 1, 2, …, 9, and so on for other digits). Convert the 
confusion matrix to an image and save it as “Confusion.jpg”. 
Important Notes: 
• Do not use Python built-in functions for mean, covariance or the 
Fisher’s linear discriminant. You have to implement your own version of 
all needed functions. You are only allowed to use functions that load 
images into Python.
• This is an individual assignment. It is not a team assignment. 
• To compute the bias term for Fisher’s linear discriminant, you can use the following 
equation:
w0 = -w^T(m1 + m2)/2
