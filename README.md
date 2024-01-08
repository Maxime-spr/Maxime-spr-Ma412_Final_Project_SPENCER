# AERO 4 - Mathematical Tools for Data Science (2023/2024)

## Project Overview

This repository contains the implementation of unsupervised clustering techniques applied to a dataset of aircraft trajectories. The goal is to group the examples in the dataset based on their features. 

## Dataset

The dataset, 'data.npy', consists of 3879 examples with 18 features each. Each example represents an aircraft trajectory with its position in the sky and other significant features. The data are clean, and no pre-processing is required.

## Repository Structure

- `code_method/`: Contains the Python source code file.
  - `choosecluster.py`: Python code searching the number of clusters of the dataset.
  - `clustervisualize.py`: Python code to visualize the clusters of the dataset.
  - `compRegression.py`: Python code whiwh compare the three regression method(Ridge, Lasso and Elastic net) to determine the most efficient.
  - `DBSCAN.py`: Python code using the DBScan clustering method on the dataset.
  - `elasticnet.py`: Python code of the Lasso regression method on the dataset.
  - `gaussiankernel.py`: Python code of the Gaussian kernel method on the dataset.
  - `GMM.py`: Python code of the Gaussian mixtures model method on the dataset.
  - `kmeans.py`: Python code of the K-means method on the dataset.
  - `lassoregression.py`: Python code of the Lasso regression method on the dataset.
  - `maxlikelihood.py`: Python code of the maximum likelihood method on the dataset.
  - `neuralnetwork.py`: Python code of the neural network method on the dataset.
  - `optics.py`: Python code of the Ordering Points To Identify the Clustering Structure on the dataset.
  - `pca.py`: Python code of the Principal component analysis method on the dataset.
  - `ridgeRgression.py`: Python code of the Ridge regression method on the dataset.
  - `svm.py`: Python code of the Support vector machine method on the dataset.
  - `utils4.py`: Library Python code of the function used for the method of svm and gaussian kernel.
  - `utils.py`: Library Python code of the function used for the other method.

- `data.npy`: The dataset file.
- `Ma412_Final_Project_SPENCER.pdf`: Document explaining the problem, possible solutions, chosen methods, and algorithm explanations.

## Getting Started

To get started with the project, follow these steps:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/Maxime-spr/Maxime-spr-Ma412_Final_Project_SPENCER.git
    ```

2. **Navigate to the Cloned Repository:**
    ```bash
    cd Maxime-spr-Ma412_Final_Project_SPENCER
    ```
