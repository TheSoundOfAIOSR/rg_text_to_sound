from typing import Callable

def explain(func: Callable) -> str:
    """returns the docstring of a function

    Args:
        func (Callable): the funcion from which to get the docstring

    Returns:
        str: the docstring of the function
    """
    return func.__doc__