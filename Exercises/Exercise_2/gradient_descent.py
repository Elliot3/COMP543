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

# Load in the gradient descent optimizing function


def gd_optimize(a):

    # Input the know parameters

    x = a
    ler_rate = 1
    threshold = 10e-20
    theta = 99999
    f_post = f(x[0], x[1])

    # Loop to perform the gradient descent

    while theta > threshold:

        x[0] = x[0] - ler_rate * grad_x(x[0], x[1])
        x[1] = x[1] - ler_rate * grad_y(x[0], x[1])

        f_prior = f_post
        f_post = f(x[0], x[1])

        theta = abs(f_post - f_prior)

        if f_post > f_prior:

            ler_rate = ler_rate * 0.5

        elif f_post < f_prior:

            ler_rate = ler_rate * 1.1

        print(f_post)

    print(x)


# Output the results

print("")
print("Results for gd_optimize(np.array([-0.2, -1.0])):")
print("")

gd_optimize(np.array([-0.2, -1.0]))

print("")
print("Results for gd_optimize(np.array([-0.5, -1.5])):")
print("")

gd_optimize(np.array([-0.5, -1.5]))
