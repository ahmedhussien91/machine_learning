import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#load Data
numpy_points_arr = np.loadtxt("SpectData.txt", delimiter='\t')


def spectral_clustering(sigma, OutputFigName):
    #calculate Laplacian(D-W) of fully connected points
    Laplacian_Matrix = []
    for idx_i, numpy_point_i in enumerate(list(numpy_points_arr)):
        Laplacian_Matrix.append([])
        for idx_j, numpy_point_j in enumerate(list(numpy_points_arr)):
            #append negative weights
            Laplacian_Matrix[idx_i].append(-math.exp(-((np.linalg.norm(numpy_point_i - numpy_point_j))**2)/(2*(sigma**2))))
        Laplacian_Matrix[idx_i][idx_i] = 0
        Laplacian_Matrix[idx_i][idx_i] = abs(sum(Laplacian_Matrix[idx_i]))

    # Calculating the eigen vector and eignen values
    Values,Vector = np.linalg.eigh(np.array(Laplacian_Matrix))

    # get 2nd min
    minPostiveValue = pd.Series(Values)[pd.Series(Values) > 0.0].min()
    IndexOfMinPostiveValue = pd.Series(Values).index[pd.Series(Values) == minPostiveValue]
    OutPutClassValues = []
    for idx, value in enumerate(Vector[:,IndexOfMinPostiveValue]):
        if value > 0:
            #specify class A
            OutPutClassValues.append(1)
        else:
            #specfiy class B
            OutPutClassValues.append(-1)

    x = [value[0] for value in numpy_points_arr]
    y = [value[1] for value in numpy_points_arr]

    plt.scatter(x,y,c=OutPutClassValues)
    plt.savefig(OutputFigName)


sigma = 0.01
spectral_clustering(sigma, "Spect_01.jpg")
sigma = 0.03
spectral_clustering(sigma, "Spect_03.jpg")
sigma = 0.05
spectral_clustering(sigma, "Spect_05.jpg")
sigma = 0.1
spectral_clustering(sigma, "Spect_1.jpg")
