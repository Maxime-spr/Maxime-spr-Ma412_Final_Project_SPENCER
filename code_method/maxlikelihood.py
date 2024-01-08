
import numpy as np
import matplotlib.pyplot as plt
import sys 
# Load the dataset
data = np.load('data.npy')

# Extract features (X) and targets (y)
X = data[:, 0].reshape(-1, 1)  
y = data[:, 1].reshape(-1, 1) 

# Apply maximum likelihood estimation
def max_lik_estimate(X, y):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # Augment features with ones
    XTX = np.dot(X_aug.T, X_aug)
    XTy = np.dot(X_aug.T, y)

    if np.linalg.cond(XTX) < 1/sys.float_info.epsilon:
        theta_ml_aug = np.linalg.solve(XTX, XTy)
    else:
        theta_ml_aug = np.dot(np.linalg.pinv(XTX), XTy)

    return theta_ml_aug

def predict_with_estimate(Xtest, theta):
    predictions = np.dot(Xtest, theta)
    return predictions

# Get maximum likelihood estimate
theta_ml_aug = max_lik_estimate(X, y)
print('theta_ml_aug',theta_ml_aug)
# Define a test set
Xtest = np.linspace(min(X), max(X), 100).reshape(-1, 1)  # 100 x 1 vector of test inputs
Xtest_aug = np.hstack([np.ones((Xtest.shape[0], 1)), Xtest])  # Augment test features

# Predict the function values at the test points using the maximum likelihood estimator
predictions_aug = predict_with_estimate(Xtest_aug, theta_ml_aug)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(X, y, '+', markersize=10, label='Training Data')
plt.plot(Xtest, predictions_aug, label='Predictions', color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


