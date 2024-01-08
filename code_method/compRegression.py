import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# Load the dataset
data = np.load('data.npy')
X = data[:, :-1]
Y = data[:, -1]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, Y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(Y_test, ridge_pred)

# Lasso Regression
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, Y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(Y_test, lasso_pred)

# Elastic Net Regression
elastic_net_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net_model.fit(X_train, Y_train)
elastic_net_pred = elastic_net_model.predict(X_test)
elastic_net_mse = mean_squared_error(Y_test, elastic_net_pred)

# Print the MSE for each method
print("Ridge MSE:", ridge_mse)
print("Lasso MSE:", lasso_mse)
print("Elastic Net MSE:", elastic_net_mse)

# Compare and print the best performer
best_method = min(enumerate([ridge_mse, lasso_mse, elastic_net_mse]), key=lambda x: x[1])[0]

if best_method == 0:
    print("Ridge Regression performs the best.")
elif best_method == 1:
    print("Lasso Regression performs the best.")
else:
    print("Elastic Net Regression performs the best.")
