import time

def timer(function):
    """Decorator for timer functions
    Usage:
    @timer
    def function(a):
        pass
    """
    
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        print(f"{function.__name__} took: {end - start:.5f} sec")
        return result
    return wrapper