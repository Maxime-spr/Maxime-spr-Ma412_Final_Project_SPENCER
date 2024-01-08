import numpy as np
from matplotlib import pyplot
import utils4

# Load the data
data = np.load('data.npy', allow_pickle=True)
X, y = data[:, :2], data[:, 2]
# Display the shape of X
print(X.shape)

pyplot.scatter(X[:,0],X[:,1])
# Plot training data
utils4.plotData(X, y)


# Try different values of C
C_values = [1, 100, 1000]

for C in C_values:
    # Train the SVM model
    model = utils4.svmTrain(X, y, C, utils4.linearKernel, 1e-3, 20)
    
    # Visualize the decision boundary
    pyplot.figure()
    pyplot.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')
    utils4.visualizeBoundaryLinear(X, y, model)
    pyplot.title(f'Decision Boundary for C={C}')

pyplot.show()

