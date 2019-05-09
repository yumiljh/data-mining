import numpy as np

EPSILON = 1e-5

# Batch Gradient Descent
class BGD:

	counter = 3

	# Only for Least_squares
	# Need to rewrite for other usage
	def error_function(self, theta, X, y):
		diff = np.dot(X, theta) - y
		m = len(X)
		return (1.0 / (2 * m)) * np.dot(np.transpose(diff), diff)

	# Only for Least_squares
	# Need to rewrite for other usage
	def gradient_function(self, theta, X, y):
		diff = np.dot(X, theta) - y
		m = len(X)
		return (1.0 / m) * np.dot(np.transpose(X),diff)

	# theta: original theta, where to start
	# X: training X
	# Y: trainnig Y
	# alpha: learning step
	def gradient_descent(self, init_theta, X, y, alpha):
		theta = init_theta
		gradient = self.gradient_function(theta, X, y)
		while not np.all(np.absolute(gradient) <= EPSILON):
			if self.counter > 0:
				print gradient
				self.counter -= 1
			theta = theta - alpha * gradient
			gradient = self.gradient_function(theta, X, y)
		return theta

# Stochastic Gradient Descent
class SGD:
	
	counter = 3

	# Only for Least_squares
	# Need to rewrite for other usage
	def error_function(self, theta, X, y):
		diff = np.dot(X, theta) - y
		m = len(X)
		return (1.0 / (2 * m)) * np.dot(np.transpose(diff), diff)

	# Only for Least_squares
	# Need to rewrite for other usage
	def gradient_function(self, theta, X, y):
		m = len(X)
		i = np.random.randint(0, m)
		diff = np.dot(X[i], theta) - y[i]
		return (1.0 / m) * np.dot(np.transpose(X), np.full(m, diff).reshape(-1,1))

	# theta: original theta, where to start
	# X: training X
	# Y: trainnig Y
	# alpha: learning step
	def gradient_descent(self, init_theta, X, y, alpha):
		theta = init_theta
		gradient = self.gradient_function(theta, X, y)
		while not np.all(np.absolute(gradient) <= EPSILON):
			if self.counter > 0:
				print gradient
				self.counter -= 1
			theta = theta - alpha * gradient
			gradient = self.gradient_function(theta, X, y)
		return theta

