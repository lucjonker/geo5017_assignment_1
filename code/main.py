import numpy as np
import matplotlib.pyplot as plt

# Define Global variables
array = np.array([[2, 0, 1],
                  [1.08, 1.68, 2.38],
                  [-0.83, 1.82, 2.49],
                  [-1.97, 0.28, 2.15],
                  [-1.31, -1.51, 2.59],
                  [0.57, -1.91, 4.32]])
t = [t for t in range(1, len(array) + 1)]
x = array[:, 0]
y = array[:, 1]
z = array[:, 2]


def visualize_position(start, dependent, independent, learning_rate, num_iterations, objective, gradient):
    plt.scatter(independent, dependent)
    f = None
    coefficients = gradient_descent(start, dependent, independent, objective, gradient, learning_rate, num_iterations)
    if len(coefficients) == 2:
        a, b = coefficients
        f = [a + (b * val) for val in independent]
    if len(coefficients) == 3:
        a0, a1, a2 = coefficients
        f = [a0 + (a1 * val) + (a2 * (val ** 2)) for val in independent]

    if f is not None:
        plt.plot(independent, f)
        # Show the plot
        plt.show()


def visualize_position_3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot(x, y, z)

    plt.show()


def objective_func_linear(dependent, independent, coefficients):
    alpha, beta = coefficients
    sum = 0
    for i, value in enumerate(dependent):
        sum += (value - (alpha + beta * independent[i])) ** 2
    return sum


def gradient_func_linear(dependent, independent, coefficients):
    alpha, beta = coefficients
    n = len(independent)

    a = (n * alpha +
         beta * np.sum(independent) -
         np.sum(dependent))
    b = (alpha * np.sum(independent) +
         beta * np.sum(np.square(independent)) -
         np.sum(np.multiply(independent, dependent)))

    return np.array([a, b])


def objective_func_polynomial(dependent, independent, coefficients):
    a0, a1, a2 = coefficients
    sum = 0
    for i, value in enumerate(dependent):
        sum += (value - (a0 + (a1 * independent[i]) + (a2 * (independent[i] ** 2)))) ** 2
    return sum


def gradient_func_polynomial(dependent, independent, coefficients):
    a0, a1, a2 = coefficients
    n = len(independent)

    da0 = (n * a0 +
           a1 * np.sum(independent) +
           a2 * np.sum(np.square(independent)) -
           np.sum(dependent))
    da1 = (a0 * np.sum(independent) +
           a1 * np.sum(np.square(independent)) +
           a2 * np.sum(np.pow(independent, 3)) -
           np.sum(np.multiply(dependent, independent)))
    da2 = (a0 * np.sum(np.square(independent)) +
           a1 * np.sum(np.pow(independent, 3)) +
           a2 * np.sum(np.pow(independent, 4)) -
           np.sum(np.multiply(dependent, np.square(independent))))

    return np.array([da0, da1, da2])


def gradient_descent(start, dependent, independent, function, gradient, learn_rate, max_iter, tol=0.0001):
    coefficients = np.array(start)
    print("\t\tinitial coefficients =", coefficients, "\t\tf(x) =",
          "{:.3f}".format(function(dependent, independent, coefficients)))
    for it in range(max_iter):
        gradient_values = gradient(dependent, independent, coefficients)
        diff = gradient_values * learn_rate
        if np.all(np.abs(diff) < tol):
            break
        coefficients = coefficients - diff
        print("iteration =", it, "\t\tcoefficients =", coefficients, "\t\tf(x) =",
              "{:.3f}".format(function(dependent, independent, coefficients)))
    return coefficients


def main():
    options = [
        "Option 1: Change learning rate",
        "Option 2: Change number of iterations",
        "Option 3: Run Simple linear regression",
        "Option 4: Run Polynomial regression",
        "Option 5: Exit"
    ]
    learning_rate = 0.001
    num_iterations = 1000

    while True:
        print("Please select an option by pressing the corresponding number key:")
        for i, option in enumerate(options, start=1):
            print(f"{i}. {option}")

        try:
            choice = int(input("Enter your choice: "))
            if choice < 1 or choice > len(options):
                print("Invalid choice. Please select a valid option.\n")
                continue

            if choice == 1:
                learning_rate = change_learning_rate(learning_rate)
            elif choice == 2:
                num_iterations = change_num_iter(num_iterations)
            elif choice == 3:
                run_simple_lr(learning_rate, num_iterations)
            elif choice == 4:
                run_polynomial_r(learning_rate, num_iterations)
            elif choice == 5:
                print("Exiting program")
                break

        except ValueError:
            print("Invalid input. Please enter a number corresponding to an option.\n")


def change_learning_rate(learning_rate):
    print("Current learning rate is", learning_rate)
    return float(input("Please enter a new learning rate (positive float): "))


def change_num_iter(num_iterations):
    print("Current number of iterations is", num_iterations)
    return int(input("Please enter a new number of iterations (positive int): "))


def run_simple_lr(learning_rate, num_iterations):
    print("Running simple linear regression with learning rate =", learning_rate, ", num_iterations =", num_iterations)
    for axis in [x, y, z]:
        visualize_position((1, 1), axis, t, learning_rate, num_iterations, objective_func_linear, gradient_func_linear)


def run_polynomial_r(learning_rate, num_iterations):
    print("Running polynomial regression with learning rate =", learning_rate, ", num_iterations =", num_iterations)
    for axis in [x, y, z]:
        visualize_position((1, 1, 1), axis, t, learning_rate, num_iterations, objective_func_polynomial,
                           gradient_func_polynomial)


if __name__ == "__main__":
    main()
