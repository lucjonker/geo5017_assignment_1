import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import false_discovery_control

# Define Global variables
array = np.array([[2, 0, 1],
                  [1.08, 1.68, 2.38],
                  [-0.83, 1.82, 2.49],
                  [-1.97, 0.28, 2.15],
                  [-1.31, -1.51, 2.59],
                  [0.57, -1.91, 4.32]])
t = [t for t in range(1, len(array) + 1)]


def visualize_positions(start, dependents, independent, learning_rate, num_iterations, tol, objective, gradient):
    print("Do you want to add a prediction based on the resulting function?")

    options = ["Option 1: Yes", "Option 2: No"]
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    choice = int(input("Enter your choice: "))

    predict = None
    if choice == 1:
        predict = input_prediction()

    figure, (px, py, pz) = plt.subplots(1, 3, figsize=(5, 5), sharey=True)
    x_dependent = dependents[:, 0]
    y_dependent = dependents[:, 1]
    z_dependent = dependents[:, 2]

    # -- Add x
    print("###################################")
    print("Calculating coefficients for x axis")
    x_predict = visualize_position(px, start, x_dependent, independent, learning_rate, num_iterations, tol, objective, gradient, predict)
    px.set_xlabel("Time (s)")
    px.set_ylabel("Position (m)")
    px.set_title("X")
    px.grid()
    px.set_axisbelow(True)
    px.tick_params(labelleft=True)

    # -- Add y
    print("###################################")
    print("Calculating coefficients for y axis")
    y_predict = visualize_position(py, start, y_dependent, independent, learning_rate, num_iterations, tol, objective, gradient, predict)
    py.set_xlabel("Time (s)")
    py.set_ylabel("Position (m)")
    py.set_title("Y")
    py.grid()
    py.set_axisbelow(True)
    py.tick_params(labelleft=True)

    # -- Add z
    print("###################################")
    print("Calculating coefficients for z axis")
    z_predict = visualize_position(pz, start, z_dependent, independent, learning_rate, num_iterations, tol, objective, gradient, predict)
    pz.set_xlabel("Time (s)")
    pz.set_ylabel("Position (m)")
    pz.set_title("Z")
    pz.grid()
    pz.set_axisbelow(True)
    pz.tick_params(labelleft=True)
    print("###################################")

    if predict is not None:
        print(f"Predicted position at time {predict}: [{x_predict}, {y_predict}, {z_predict}]")

    subtitle = f"Initial learning rate: {learning_rate}    Max. number of iterations: {num_iterations}    Gradient tolerance: {tol}"
    figure.text(0.5, 0.92, subtitle, transform=figure.transFigure, horizontalalignment='center')

    if len(start) == 2:
        figure.suptitle("Linear regression", fontweight='bold', fontsize=20)

    if len(start) == 3:
        figure.suptitle("Polynomial regression", fontweight='bold', fontsize=20)

    plt.show()

    print("Do you want to visualize the path of the drone in 3D?")

    options = ["Option 1: Yes", "Option 2: No"]
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    choice = int(input("Enter your choice: "))

    if predict is not None and choice == 1:
        visualize_position_3d_prediction(x_dependent, y_dependent, z_dependent, x_predict, y_predict, z_predict)

    elif predict is None and choice == 1:
        visualize_position_3d(x_dependent, y_dependent, z_dependent)


def visualize_position(plot, start, dependent, independent, learning_rate, num_iterations, tol, objective, gradient,
                       predict):
    plot.scatter(independent, dependent)
    f = None
    prediction = None
    coefficients = gradient_descent(start, dependent, independent, objective, gradient, learning_rate, num_iterations, tol)
    if len(coefficients) == 2:
        a, b = coefficients
        f = [a + (b * val) for val in independent]

        if predict is not None:
            prediction = a + (b * predict)
            plot.scatter(predict, prediction)

    if len(coefficients) == 3:
        a0, a1, a2 = coefficients
        f = [a0 + (a1 * val) + (a2 * (val ** 2)) for val in independent]

        if predict is not None:
            prediction = a0 + (a1 * predict) + (a2 * (predict ** 2))
            plot.scatter(predict, prediction)


    if f is not None:
        plot.plot(independent, f, color='r')
        if predict is not None:
            plot.legend(['Observation', 'Prediction','Trend'], loc='upper left')
        else:
            plot.legend(['Observation','Trend'], loc='upper left')

        return prediction

def visualize_position_3d_prediction(x, y, z, xp, yp, zp):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    ax.scatter(xp, yp, zp)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot(x, y, z, color='r')
    ax.plot([x[-1],xp], [y[-1],yp], [z[-1],zp], color='orange')

    ax.legend(['Observation', 'Prediction', 'Trend'], loc='upper left')
    ax.set_title("Trajectory of drone")
    plt.show()

def visualize_position_3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot(x, y, z, color='red')
    ax.legend(['Observation', 'Trend'], loc='upper left')
    ax.set_title("Trajectory of drone")
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


def gradient_descent(start, dependent, independent, function, gradient, learn_rate, max_iter, tol):
    coefficients = np.array(start)
    error = function(dependent, independent, coefficients)
    num_iterations = max_iter
    print("Initial coefficients =", coefficients, "\t\t initial learn rate =", learn_rate, "\t\tinitial residual error f(x) =", "{:.3f}".format(error))

    for it in range(max_iter):
        descent = False
        tolerance = False

        while not descent:
            gradient_values = gradient(dependent, independent, coefficients)
            diff = gradient_values * learn_rate
            if np.all(np.abs(diff) < tol):
                tolerance = True
                num_iterations  = it
                break

            new_coefficients = coefficients - diff
            new_error = function(dependent, independent, new_coefficients)

            if new_error < error:
                descent = True
                coefficients = new_coefficients
            else:
                learn_rate = learn_rate * 0.25

            error = new_error

        if tolerance:
            break

    print("final iteration =", num_iterations, "\t\t learn rate =", learn_rate, "\t\tcoefficients =", coefficients,
          "\t\tfinal residual error f(x) =",
          "{:.3f}".format(function(dependent, independent, coefficients)))
    return coefficients


def main():
    options = [
        "Option 1: Change initial learning rate",
        "Option 2: Change number of iterations",
        "Option 3: Change gradient tolerance",
        "Option 4: Run Simple linear regression",
        "Option 5: Run Polynomial regression",
        "Option 6: Exit"
    ]
    learning_rate = 0.1
    tol = 0.00001
    num_iterations = 10000

    while True:
        print("\nPlease select an option by pressing the corresponding number key:")
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
                tol = change_gradient_tol(tol)
            elif choice == 4:
                run_simple_lr(learning_rate, num_iterations, tol)
            elif choice == 5:
                run_polynomial_r(learning_rate, num_iterations, tol)
            elif choice == 6:
                print("Exiting program")
                break

        except ValueError:
            print("Invalid input. Please enter a number corresponding to an option.\n")


def change_learning_rate(learning_rate):
    print("Current initial learning rate is", learning_rate)
    return float(input("Please enter a new initial learning rate (positive float): "))


def change_num_iter(num_iterations):
    print("Current number of iterations is", num_iterations)
    return int(input("Please enter a new number of iterations (positive int): "))


def change_gradient_tol(tol):
    print("Current gradient tolerance is", tol)
    return float(input("Please enter a new gradient tolerance (positive float): "))

def input_prediction():
    prediction = input("Please enter at which timestep you want your prediction (positive int): ")

    try:
        p = int(prediction)

    except ValueError:
        print("Please enter a valid value")
        p = input_prediction()

    return p


def run_simple_lr(learning_rate, num_iterations, tol):
    print("Running simple linear regression with initial learning rate =", learning_rate, ", num_iterations =", num_iterations, ", tolerance =", tol)

    visualize_positions((1, 1), array, t, learning_rate, num_iterations, tol, objective_func_linear, gradient_func_linear)


def run_polynomial_r(learning_rate, num_iterations, tol):
    print("Running polynomial regression with initial learning rate =", learning_rate, ", num_iterations =", num_iterations, ", tolerance =", tol)
    visualize_positions((1, 1, 1), array, t, learning_rate, num_iterations, tol, objective_func_polynomial,
                        gradient_func_polynomial)


if __name__ == "__main__":
    main()
