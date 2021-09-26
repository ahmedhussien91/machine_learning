# Problem
Implement the spectral clustering algorithm that uses the sign of the components of the 
eigenvector corresponding to the second smallest eigenvalue for clustering. Use the 
following weight function:
			w = exp(-||xi -xj||^2/2*sigma^2
Apply the clustering algorithm to the 2-dimension data provided in the file “SpectData.txt” 
with the goal of identifying two clusters corresponding to two concentric circles. Each row 
in the file corresponds to one data point. 
Deliverables:
• Your code.
• A plot of the points provided in the dataset after clustering showing the two 
identified clusters using σ = 0.01. Name your plot “Spect_01.jpg”.
• A plot of the points provided in the dataset after clustering showing the two 
identified clusters using σ = 0.05. Name your plot “Spect_05.jpg”.
• A plot of the points provided in the dataset after clustering showing the two 
identified clusters using σ = 0.1. Name your plot “Spect_1.jpg”.
