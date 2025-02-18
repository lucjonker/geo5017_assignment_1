import numpy as np
import matplotlib.pyplot as plt

array = np.array([[2,0,1],
                  [1.08,1.68,2.38],
                  [-0.83, 1.82, 2.49],
                  [-1.97, 0.28, 2.15],
                  [-1.31,-1.51,2.59],
                  [0.57,-1.91,4.32]])
t = [t for t in range(len(array))]
x = array[:, 0]
y = array[:, 1]
z = array[:, 2]

def visualize_position(dependent,t):
    plt.scatter(t, dependent)

    a, b = our_gradient_descent([5, 5], objective_func, gradient, 0.01, 100, t, dependent)

    f = [a+b*dependent for dependent in t]

    print("The speed of the drone is", b)

    plt.plot(t, f)

    # Show the plot
    plt.show()

def visualize_position_3d(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot(x,y,z)

    plt.show()

def objective_func(dependent, time, alpha, beta):
    sum = 0
    for i, value in enumerate(dependent):
        sum += (value -  (alpha + beta*time[i]))**2
    return sum

def gradient(alpha, beta, dependent, time):
    n = len(time)

    a = n * alpha + beta * np.sum(time) - np.sum(dependent)
    b = alpha * np.sum(x) + beta * np.sum(np.square(time)) - np.sum(np.multiply(time,dependent))

    return a, b

def our_gradient_descent(start, function, gradient, learn_rate, max_iter, independent, dependent ,tol=0.001):
    alpha_g , beta_g = start # -- Initial guess
    for it in range(max_iter):
        alpha_grad, beta_grad = gradient(alpha_g, beta_g, independent, dependent)
        a_diff = learn_rate * alpha_grad
        b_diff = learn_rate * beta_grad
        if np.abs(a_diff)  < tol and np.abs(b_diff) < tol:
            break
        print("iteration =", it, "\t\talpha =", "{:.5f}".format(alpha_g), "\t\tbeta =","{:.5f}".format(beta_g), "\t\tf(x) =", "{:.3f}".format(function(dependent, independent, alpha_g, beta_g)))
        alpha_g = alpha_g - a_diff  # --  Update the current point
        beta_g = beta_g - b_diff
    return alpha_g, beta_g

#our_gradient_descent([5,5],objective_func,gradient, 0.01, 100, t, z)
visualize_position(z,t)



