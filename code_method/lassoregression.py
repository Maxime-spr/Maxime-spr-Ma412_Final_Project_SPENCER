import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = np.load('data.npy')
X = data[:, :10]  # Features
Y = data[:, 10]   # Target variable


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define a range of alpha values
alphas = np.logspace(-3, 3, num=100)

# Perform LASSO regression to find the best alpha value
lasso_cv_model = LassoCV(alphas=alphas, cv=5)
lasso_cv_model.fit(X_train, Y_train)

best_alpha = lasso_cv_model.alpha_

# Fit a LASSO regression model with the best alpha on the entire dataset
lasso_best_model = Lasso(alpha=best_alpha)
lasso_best_model.fit(X, Y)

# Make predictions on the entire dataset
Y_pred = lasso_best_model.predict(X)

# Compute the mean squared error
mse = mean_squared_error(Y, Y_pred)

print("Best Lambda Value:", best_alpha)
print("Mean Squared Error (MSE) on Entire Dataset:", mse)

# Plot the evolution of coefficients with LASSO regularization
coefficients = []
for alpha in alphas:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X, Y)
    coefficients.append(lasso_model.coef_)

coefficients = np.array(coefficients)

plt.figure(figsize=(12, 6))
for i in range(X.shape[1]):
    plt.plot(alphas, coefficients[:, i], label=f'Feature {i+1}')

plt.xscale('log')
plt.xlabel('Regularization Parameter (λ)')
plt.ylabel('Coefficient Value (𝛼̂)')
plt.title('Coefficient Evolution with LASSO Regularization')
plt.legend()
plt.grid()
plt.show()
