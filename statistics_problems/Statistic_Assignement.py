import numpy as np
import matplotlib.pyplot as plot
import math
from scipy.stats import norm

#Question 1
def Q1_Function():
    #load Data from File
    np_array = np.loadtxt('./Assignment Dataset/Data1.txt')
    #sorting Data
    np_array.sort()
    #calculating Q2
    q2 = np.median(np_array)
    q2_index_arr = np.argwhere(np.isclose(np_array,q2))
    q2_index = q2_index_arr[int(len(q2_index_arr)/2)]
    #calculating Q1
    q1 = np.median(np_array[1:q2_index[0]+1])
    q1_index_arr = np.argwhere(np.isclose(np_array,q1))
    q1_index = q1_index_arr[int(len(q1_index_arr)/2)]
    #calculating Q3
    q3 = np.median(np_array[q2_index[0]+1:])
    q3_index_arr = np.argwhere(np.isclose(np_array,q3))
    q3_index = q3_index_arr[int(len(q3_index_arr)/2)]
    #calculating inter-quartile range
    IQR = q3 - q1
    #outlier points
    OutlierPoints = np_array[np.where((np_array[:] < (q1 - 1.5*IQR)) | (np_array[:] >(q3 + 1.5*IQR)))]
    #extreme outlier points
    ExtreamOutlierPoints = np_array[np.where((np_array < (q1 - 3*IQR)) | (np_array >(q3 + 3*IQR)))]
    #print Qutput
    print("q1 = ",q1)
    print("q2 = ",q2)
    print("q3 = ",q3)
    print("IQR = q3-q1 = ", q3-q1)
    print("OutlierPoints = ", OutlierPoints)
    print("There are no extreme outlier points")
    #Box Plot the Data
    plot.boxplot(np_array)
    plot.show()

    return
#Question 2
def Q2_function():
    #simulate the throwing of Die 1000 times
    DieThrowingOutput_arr = np.random.uniform(1,7,1000).astype('uint8')
    #plot Histogram
    plot.hist(DieThrowingOutput_arr, normed=True, bins=100)
    plot.ylabel('Probability')
    plot.show()
    #simulate the throwing of Die 1000 times twice
    DieThrowingOutput_arr1 = np.random.uniform(1,7,1000).astype('uint8')
    DieThrowingOutput_arr2 = np.random.uniform(1,7,1000).astype('uint8')
    DieThrowingOutput_avg = (DieThrowingOutput_arr2 + DieThrowingOutput_arr1)/2
    #plot Histogram
    plot.hist(DieThrowingOutput_avg, normed=True, bins=100)
    plot.ylabel('Probability')
    plot.show()
    #Calculate Mean and Variance
    print("Mean = ", DieThrowingOutput_avg.mean())
    print("Variance = ", DieThrowingOutput_avg.var())
    #simulate the throwing of Die 1000 times twice
    DieThrowingOutput_avg = np.zeros(1000)
    for i in range(0,10):
        DieThrowingOutput_arr1 = np.random.uniform(1,7,1000).astype('uint8')
        DieThrowingOutput_avg += DieThrowingOutput_arr1
    DieThrowingOutput_avg = DieThrowingOutput_avg/10
    #plot Histogram
    plot.hist(DieThrowingOutput_avg, normed=True, bins=100)
    plot.ylabel('Probability')
    plot.show()
    #Calculate Mean and Variance
    print("Mean = ", DieThrowingOutput_avg.mean())
    print("Variance = ", DieThrowingOutput_avg.var())
    return

#Question 3
def compute_p_value(significance_level, DataSet1, DataSet2):
    # Ho -> The mean of dataset1 == the mean of dataset2
    # H1 -> The mean of dataset1 != the mean of dataset2
    # P_value = P()
    p_value = 0
    boundaries = 0
    mean1 = DataSet1.mean()
    std1 = DataSet1.std()
    mean2 = DataSet2.mean()
    Zo = (mean2 - mean1)/ (std1/math.sqrt(len(DataSet2)))
    p_value = 2*(1-norm.cdf(Zo))

    Zo_boundaries = norm.ppf(significance_level/2)
    boundaries = [mean1 + (Zo_boundaries* (std1/math.sqrt(len(DataSet2)))), mean1 - (Zo_boundaries* (std1/math.sqrt(len(DataSet2))))]
    # ploting Histogram of the two data sets
    plot.hist(DataSet1, normed=True, bins=100)
    plot.hist(DataSet2, normed=True, bins=100)
    plot.ylabel('Probability')
    plot.show()
    rejectNull = True if (p_value < significance_level) else False

    return [rejectNull, p_value, boundaries]


#Question 1
Q1_Function()
#Question 2
Q2_function()
#Question 3
#load Data Sets
nparr_DataSet3_1 = np.loadtxt('./Assignment Dataset/Data3-1.txt', delimiter=',') # Population
nparr_DataSet3_2 = np.loadtxt('./Assignment Dataset/Data3-2.txt', delimiter=',') # sample 1
nparr_DataSet3_3 = np.loadtxt('./Assignment Dataset/Data3-3.txt', delimiter=',') # sample 2
rejectNull, p_value, boundaries = compute_p_value(0.05, nparr_DataSet3_1, nparr_DataSet3_2)
print("p_value = ",p_value)
if rejectNull:
    print("We Reject Null Hyposis P_Value < Significance level")
else:
    print("We Accept Null Hyposis P_Value > Significance level")
print("boundaries = ",boundaries)
rejectNull, p_value, boundaries = compute_p_value(0.05, nparr_DataSet3_1, nparr_DataSet3_3)
print("p_value = ",p_value)
if rejectNull:
    print("We Reject Null Hyposis P_Value < Significance level")
else:
    print("We Accept Null Hyposis P_Value > Significance level")
print("boundaries = ",boundaries)
