import numpy as np

EPSILON = 1e-5

# Only for Least_squares
# Need to rewrite for other usage
def error_function(theta, X, y):
	diff = np.dot(X, theta) - y
	m = len(X)
	return (1.0 / (2 * m)) * np.dot(np.transpose(diff), diff)

# Batch Gradient Descent	
# Only for Least_squares
# Need to rewrite for other usage
def BGD_gradient_function(theta, X, y):
	diff = np.dot(X, theta) - y
	m = len(X)
	return (1.0 / m) * np.dot(np.transpose(X),diff)

# Stochastic Gradient Descent
# Only for Least_squares
# Need to rewrite for other usage
def SGD_gradient_function(theta, X, y):
	m = len(X)
	i = np.random.randint(0, m)
	diff = np.dot(X[i], theta) - y[i]
	return (np.transpose(X[i]) * diff).reshape(-1,1)

# theta: original theta, where to start
# X: training X
# Y: trainnig Y
# alpha: learning step
# BGD_SGD: True for SGD, False for BGD
def gradient_descent(init_theta, X, y, alpha, BGD_SGD = False):
	theta = init_theta
	gradient_function = lambda theta, X, y: SGD_gradient_function(theta, X, y) if BGD_SGD else BGD_gradient_function(theta, X, y)
	gradient = gradient_function(theta, X, y)
	while not np.all(np.absolute(gradient) <= EPSILON):
		theta = theta - alpha * gradient
		gradient = gradient_function(theta, X, y)
	return theta
