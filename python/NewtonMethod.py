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
    # Ask the user to choose between the two functions
    choice = input("Choose the function to use (1 or 2):\n"
                   "1. f(x) = x^2 - x - 1\n"
                   "2. f(x) = x^3 - x^2 - 2*x + 1\n"
                   "Enter your choice (1 or 2): ")

    # Assign the selected function and its derivative
    if choice == '1':
        func = lambda x: x**2 - x - 1
        grad = lambda x: 2*x - 1
    elif choice == '2':
        func = lambda x: x**3 - x**2 - 2*x + 1
        grad = lambda x: 3*x**2 - 2*x - 2
    else:
        print("Invalid choice! Please enter 1 or 2.")
        exit()

    # Initial guesses for the roots
    initial_guesses = [-1, 1, 2.5]

    # Plot the function over a broad interval for visualization
    plot_function(func, -2, 3)

    # Loop through each initial guess
    for i, x0 in enumerate(initial_guesses, start=1):
        
        # Call the Newton's method for the root with the given initial guess
        our_root = newton_method(func, grad, x0)
        
        # Call SciPy method (reference method) for the root with the given initial guess
        sp_result = sp.optimize.root(func, x0)
        sp_root = sp_result.x.item()
        
        # Print the result for this initial guess
        print(f"\nRoot found starting from x0 = {x0} by Newton's Method = {our_root:0.8f}.")
        print(f"Root found starting from x0 = {x0} by SciPy = {sp_root:0.8f}.\n")

    print("Root finding process completed for all initial guesses.")
