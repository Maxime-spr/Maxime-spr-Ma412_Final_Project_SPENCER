# AERO 4 - Mathematical Tools for Data Science (2023/2024)

## Project Overview

This repository contains the implementation of unsupervised clustering techniques applied to a dataset of aircraft trajectories. The goal is to group the examples in the dataset based on their features. 

## Dataset

The dataset, 'data.npy', consists of 3879 examples with 18 features each. Each example represents an aircraft trajectory with its position in the sky and other significant features. The data are clean, and no pre-processing is required.

## Repository Structure

- `code_method/`: Contains the Python source code file.
  - `choosecluster.py`: Python code searching the number of clusetrs of the dataset.
  - `clustervisualize.py`: Python code to visualize the clusters of the dataset.
  - `compRegression.py`: Python code whiwh compare the three regression method(Ridge, Lasso and Elastic net) to determine the most efficient.
  - `DBSCAN.py`: Python code using the DBScan clustering method on the dataset.
  - `elasticnet.py`: Python code of the Lasso regression method on the dataset.
  - `gaussiankernel.py`: Python code of the Gaussian kernel method on the dataset.
  - `GMM.py`: Python code of the Gaussian mixtures model method on the dataset.
  - `kmeans.py`: Python code of the K-means method on the dataset.
  - `lassoregression.py`: Python code of the Lasso regression method on the dataset.
- `data.npy`: The dataset file.
- `Project_Presentation.pdf`: Document explaining the problem, possible solutions, chosen methods, and algorithm explanations.
