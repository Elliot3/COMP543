# Load in the necessary libraries

import numpy as np
import math

# Load in the McCormick function


def f(x, y):

    return math.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

# Load in the functions for gradients in x and y directions


def grad_x(x, y):

    return math.cos(x + y) + 2 * (x - y) - 1.5


def grad_y(x, y):

    return math.cos(x + y) - 2 * (x - y) + 2.5

# Load in the functions for the second derivatives


def grad_xx(x, y):

    return -math.sin(x + y) + 2


def grad_xy(x, y):

    return -math.sin(x + y) - 2


def grad_yy(x, y):

    return -math.sin(x + y) + 2

# Load in the function to create the hessian matrix


def create_hessian(x, y):

    return np.array([[grad_xx(x, y), grad_xy(x, y)], [grad_xy(x, y), grad_yy(x, y)]])

# Load the Newton's method optimization function


def nm_optimize(a):

    # Input the know parameters

    x = a
    f_post = f(x[0], x[1])
    theta = 99999

    # Loop to perform the Newton's method

    while theta != 0:

        der_loss = np.array([grad_x(x[0], x[1]), grad_y(x[0], x[1])])
        x = x - np.dot(np.linalg.inv(create_hessian(x[0], x[1])), der_loss)

        f_prior = f_post
        f_post = f(x[0], x[1])

        theta = abs(f_prior - f_post)

        print(f_post)

    print(x)


# Output the results

print("")
print("Results for nm_optimize(np.array([-0.2, -1.0])):")
print("")

nm_optimize(np.array([-0.2, -1.0]))

print("")
print("Results for nm_optimize(np.array([-0.5, -1.5])):")
print("")

nm_optimize(np.array([-0.5, -1.5]))
