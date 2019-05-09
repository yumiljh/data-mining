# coding:utf-8
from gradient_descent import *
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1,21).reshape(-1,1)
X = np.column_stack((np.ones(20).reshape(-1,1), x))
y = np.array([3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
		    11, 13, 13, 16, 17, 18, 17, 19, 21]).reshape(-1,1)
'''
x = np.array([[ 1.19304077],[ 0.94091142],[ 0.02339566],[-0.8981732 ],[ 0.09943588],[ 0.00383871],[-0.80779895],[ 0.87421378],[ 2.3114854 ],[ 0.76986146],[ 0.05000725],[ 1.58109966],[ 2.82690925],[-0.96564311],[-0.16234562],[ 0.43630434],[ 0.76292887],[-2.69115914],[-1.2516014 ],[-1.65755253]])
X = np.column_stack((np.ones(len(x)).reshape(-1,1),x))
y = np.array([30.16130442,36.19745451,17.10101282,-7.65571261,26.76324042,-5.03382704,-21.33951587,22.67316442,54.01695548,23.33351057,2.82356836,25.49128867,91.31885123,-25.95196596,10.49228655,4.74371435,4.69407196,-90.81921826,-13.93506075,-34.20976885]).reshape(-1,1)
'''

alpha = 0.001

# h_theta(x_i) = theta_0 + theta_1 * x1_i
theta = np.array([1,1]).reshape(-1,1)

theta = BGD().gradient_descent(theta, X, y, alpha)
print "BGD:", theta
print BGD().error_function(theta, X, y)

plt.figure(1)
plt.scatter(x,y,color= 'gray')
plt.plot(x,x*theta[1]+theta[0],color='blue',linewidth=3)

theta2 = np.array([1,1]).reshape(-1,1)

theta2 = SGD().gradient_descent(theta2, X, y, alpha)
print "SGD:", theta2
print SGD().error_function(theta2, X, y)

plt.figure(2)
plt.scatter(x,y,color='gray')
plt.plot(x,x*theta2[1]+theta2[0],color='orange',linewidth=3)

plt.show()

