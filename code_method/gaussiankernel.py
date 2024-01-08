import numpy as np
from matplotlib import pyplot
import utils4

#gaussian kernel
def gaussianKernel(x1, x2, sigma):

    sim = 0
    # ============================================================
    distance = np.sum((x1 - x2) ** 2)
    
    sim = np.exp(-distance / (2 * (sigma ** 2)))
    # ============================================================
    return sim

# Load the data
data = np.load('data.npy', allow_pickle=True)
X, y = data[:, :2], data[:, 2]

# Choose two data points
x1 = X[0, :]
x2 = X[1, :]
sigma = 1  # You can adjust the sigma value

# Test the Kernel
sim = gaussianKernel(x1, x2, sigma)

print(f'Gaussian Kernel between x1 = {x1}, x2 = {x2}, sigma = {sigma}:')
print(f'\t{sim}\n')

# Try different values of C and sigma
C_values = [0.1]
sigma_values = [1]

for C in C_values:
    for sigma in sigma_values:
        model = utils4.svmTrain(X, y, C, lambda x1, x2: gaussianKernel(x1, x2, sigma), 1e-3, 20)

        pyplot.figure()
        utils4.visualizeBoundary(X, y, model)
        pyplot.title(f'SVM Decision Boundary with Gaussian Kernel (C={C}, Sigma={sigma})')


pyplot.show()



# Plot training data
pyplot.scatter(X[:, 0], X[:, 1], c=y, cmap='jet')
pyplot.title('Training Data')
pyplot.xlabel('X1')
pyplot.ylabel('X2')
pyplot.show()

# Split the data into training and validation sets
# Assuming you want to use 80% for training and 20% for validation
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X, Xval = X[:split_index], X[split_index:]
y, yval = y[:split_index], y[split_index:]


def BestParams(X, y, Xval, yval):
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    best_C = None
    best_sigma = None
    best_error = float('inf')

    for C in C_values:
        for sigma in sigma_values:
            model = utils4.svmTrain(X, y, C, lambda x1, x2: gaussianKernel(x1, x2, sigma), 1e-3, 20)

            predictions = utils4.svmPredict(model, Xval)

            error = np.mean(predictions != yval) * 100

            if error < best_error:
                best_error = error
                best_C = C
                best_sigma = sigma

    return best_C, best_sigma

best_C, best_sigma = BestParams(X, y, Xval, yval)

print(f"Best C: {best_C}")
print(f"Best Sigma: {best_sigma}")

C, sigma = BestParams(X, y, Xval, yval);
model= utils4.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils4.visualizeBoundary(X, y, model)
print('C = ',C,'sigma = ',sigma)