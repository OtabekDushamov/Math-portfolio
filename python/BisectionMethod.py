# Root Finding Method
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plot_function(func, a, b):
    """
    This function plots the graph of the input function 
    within the given interval [a, b).
    """
    x = np.linspace(a, b, 400)
    y = func(x)
    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.title('Plot of the function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()


def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method to find the root of a function within a given interval.

    Parameters:
    - func: The function for which the root is to be found.
    - a, b: Interval [a, b] within which the root is searched for.
    - tol: Tolerance level for checking convergence of the method.
    - max_iter: Maximum number of iterations.

    Returns:
    - root: Approximation of the root.
    
    Example
    --------
    >>> fun = lambda x: x**2 - x - 1
    >>> root = bisection_method(fun, 1, 2, max_iter=20)
    """

    # Check if the interval is valid (signs of f(a) and f(b) are different)
    if func(a) * func(b) >= 0:
        print("Bisection method fails. The function must have different signs at a and b.")
        return None

    # Main loop starts here
    iter_count = 1
    while iter_count <= max_iter:
        # Midpoint
        c = (a + b) / 2
        # Check if the midpoint is the root or close enough
        if abs(func(c)) <= tol or (b - a) / 2 < tol:
            return c

        # Determine the subinterval to continue with
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
        
        iter_count += 1
    
    print("Warning! Exceeded the maximum number of iterations.")
    return c

# Example usage:
if __name__ == "__main__":

    while True:
        # Ask the user to choose the function
        func_choice = input("Enter '1' for the first function (x^2 - x - 1) or '2' for the second function (x^3 - x^2 - 2x + 1): ")
        if func_choice == '1':
            func = lambda x: x**2 - x - 1  # First Function
            plot_function(func, -2, 3)

            # Set the intervals for the roots
            intervals = [(0, 2), (-2, 0)]
            break
        
        elif func_choice == '2':
            func = lambda x: x**3 - x**2 - 2*x + 1  # Second Function
            plot_function(func, -2, 3)

            # Set the intervals for the roots
            intervals = [(-2, -1), (0, 1), (1, 2)]
            break
        
        else:
            print("Invalid input. Please enter '1' or '2'.")

    # Finding roots using the Bisection Method
    roots = []
    for (a, b) in intervals:
        root = bisection_method(func, a, b)
        if root is not None:
            roots.append(root)

    # Finding roots using SciPy's method for comparison
    scipy_roots = []
    for (a, b) in intervals:
        x0 = (a + b) / 2
        sp_result = sp.optimize.root(func, x0)
        sp_root = sp_result.x.item()
        scipy_roots.append(sp_root)

    # Print the results
    for i, (root, sp_root) in enumerate(zip(roots, scipy_roots), 1):
        print(f"Root {i} found by Bisection Method = {root:.8f}.")
        print(f"Root {i} found by SciPy = {sp_root:.8f}.")
