import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = np.load('data.npy')

# Assuming the dataset has 10 features and a target variable (adjust accordingly)
X = data[:, :10]  # Features
Y = data[:, 10]   # Target variable

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Ridge Regression function
def ridge_regression(X, Y, alpha):
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X, Y)
    
    beta = ridge_model.coef_
    alpha = ridge_model.intercept_
    
    return alpha, beta

# Ridge Regression with lambda = 1.0
lambda_value = 1.0
alpha_ridge, beta_ridge = ridge_regression(X_train, Y_train, alpha=lambda_value)

# Print the results
print("Ridge Regression - Intercept (Alpha):", alpha_ridge)
print("Ridge Regression - Coefficients (Beta):", beta_ridge)

# Plot the evolution of the coefficients with different regularization parameters
lambda_values = np.logspace(-3, 3, num=100)
alpha_values = []

for lambda_val in lambda_values:
    ridge_model = Ridge(alpha=lambda_val)
    ridge_model.fit(X, Y)
    alpha_values.append(ridge_model.coef_)

alpha_values = np.array(alpha_values)

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(X.shape[1]):
    plt.plot(lambda_values, alpha_values[:, i], label=f'Feature {i+1}')

plt.xscale('log')
plt.xlabel('Regularization Parameter (λ)')
plt.ylabel('Coefficient Value (𝛼̂)')
plt.title('Coefficient Evolution with Ridge Regularization')
plt.legend()
plt.grid()
plt.show()

# Ridge Regression with RidgeCV
alphas = np.logspace(-3, 3, num=100)  
ridge_cv_model = RidgeCV(alphas=alphas, store_cv_values=True)

ridge_cv_model.fit(X_train, Y_train)

best_alpha = ridge_cv_model.alpha_

ridge_best_model = Ridge(alpha=best_alpha)
ridge_best_model.fit(X, Y)

Y_pred = ridge_best_model.predict(X)

mse = mean_squared_error(Y, Y_pred)

print("Best Lambda Value:", best_alpha)
print("Mean Squared Error (MSE) on Entire Dataset:", mse)
