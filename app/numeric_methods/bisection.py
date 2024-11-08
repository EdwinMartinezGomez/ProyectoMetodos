def bisection_method(equation, a=None, b=None, tol=1e-6, max_iter=100):
    """
    Implements the bisection method to find roots of an equation.
    
    Args:
        equation: Lambda function of the equation to solve
        a: Lower bound of the interval
        b: Upper bound of the interval
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    
    Returns:
        dict: Contains solution information including root, iterations, and error message if any
    """
    # Input validation
    if a is None or b is None:
        return {
            "success": False,
            "error": "Both interval bounds (a and b) must be provided"
        }
    
    try:
        # Evaluate function at endpoints
        fa = equation(a)
        fb = equation(b)
        
        # Check if either point is a root
        if abs(fa) < tol:
            return {"success": True, "root": a, "iterations": 0}
        if abs(fb) < tol:
            return {"success": True, "root": b, "iterations": 0}
        
        # Check if there's a root in the interval
        if fa * fb >= 0:
            return {
                "success": False,
                "error": "No root exists in the given interval or multiple roots exist"
            }
        
        # Bisection iteration
        iterations = 0
        while iterations < max_iter:
            c = (a + b) / 2
            fc = equation(c)
            
            if abs(fc) < tol:
                return {
                    "success": True,
                    "root": c,
                    "iterations": iterations + 1,
                    "error": 0  # Error is zero since we found the root
                }
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
                
            iterations += 1
        
        estimated_error = abs(b - a) / 2  # Estimación del error
        return {
            "success": True,
            "root": c,
            "iterations": iterations,
            "error": estimated_error
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"An error occurred: {str(e)}"
        }
