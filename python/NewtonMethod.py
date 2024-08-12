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
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Plot of the function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()


def newton_method(func, grad, x0, tol=1e-6, max_iter=100):
    '''Approximate solution of f(x)=0 by Newton-Raphson's method.

        Parameters
        ----------
        func : function 
            Function value for which we are searching for a solution f(x)=0,
        grad: function
            Gradient value of function f(x)
        x0 : number
            Initial guess for a solution f(x)=0.
        tol : number
            Stopping criteria is abs(f(x)) < tol.
        max_iter : integer
            Maximum number of iterations of Newton's method.

        Returns
        -------
        xn : root

        Example
        --------
        >>> fun = lambda x: x**2 - x - 1
        >>> grad = lambda x: 2*x - 1
        >>> root = newton_method(fun, grad, 1, max_iter=20)
        '''
    xn = x0
    iter_count = 1
    while iter_count <= max_iter:
        fxn = func(xn)
        if abs(fxn) < tol:
            return xn
        grad_xn = grad(xn)
        if grad_xn == 0:
            print("Warning! Zero derivative. No solution found.")
            return None
        xn = xn - fxn / grad_xn
        iter_count += 1

    print("Warning! Exceeded the maximum number of iterations.")
    return xn


# Main Driver Function:
if __name__ == "__main__":
    # Define the 1st Function for which the root is to be found
    func = lambda x: x**2 - x - 1
    # Define the gradient of the Function
    grad = lambda x: 2*x - 1

    # Uncomment the next two lines to use the 2nd Function
    #func = lambda x: x**3 - x**2 - 2*x + 1
    #grad = lambda x: 3*x**2 - 2*x - 2

    # Call plot_function to plot graph of the function
    plot_function(func, -2, 3)

    # Set the initial guesses for the roots
    x0_1 = 2  # Initial guess for 1st root
    x0_2 = -1  # Initial guess for 2nd root

    # Call the Newton's method for 1st root
    our_root_1 = newton_method(func, grad, x0_1)

    # Call SciPy method (reference method) for 1st root
    sp_result_1 = sp.optimize.root(func, x0_1)
    sp_root_1 = sp_result_1.x.item()

    # Call the Newton's method for 2nd root
    our_root_2 = newton_method(func, grad, x0_2)

    # Call SciPy method (reference method) for 2nd root
    sp_result_2 = sp.optimize.root(func, x0_2)
    sp_root_2 = sp_result_2.x.item()

    # Print the result
    print("1st root found by Newton's Method = {:0.8f}.".format(our_root_1))
    print("1st root found by SciPy = {:0.8f}".format(sp_root_1))

    print("2nd root found by Newton's Method = {:0.8f}.".format(our_root_2))
    print("2nd root found by SciPy = {:0.8f}".format(sp_root_2))
