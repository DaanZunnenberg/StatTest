import functools

def prelim_raise_TypeError(func) -> ...:
    functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except TypeError as e:
            raise TypeError("Invalid parameters, try using the default configuration.")
    return wrapper
