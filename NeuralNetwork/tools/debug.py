import time
import numpy as np

# Activates the trace for debugging:
DEBUG = False


def trace(debug=DEBUG):
    # Colours for the trace:
    RED = '\33[31m'
    GREEN = '\33[32m'
    YELLOW = '\33[33m'
    BLUE = '\33[34m'
    VIOLET = '\33[35m'
    BEIGE = '\33[36m'
    END = "\x1b[0m"

    def parameterized(func):
        NAME = func.__qualname__ if func.__name__ != func.__qualname__ else f"{func.__module__}.{func.__name__}"

        def wrap(*args, **kwargs):
            wrap.calls += 1
            if debug:
                # Log the function name and arguments:
                argsname_ = func.__code__.co_varnames
                args_ = ", ".join([f"{k}={VIOLET}Array{v.shape}{END}" if isinstance(v, np.ndarray) else f"{k}={YELLOW}{v}{END}" \
                                   for k, v in zip(argsname_, args) if k != 'self'])
                kwargs_ = ", ".join([f"{k}={VIOLET}Array{v.shape}{END}" if isinstance(v, np.ndarray) else f"{k}={YELLOW}{v}{END}" \
                                     for k, v in kwargs.items()])
                sep_ = ", " if args_ and kwargs_ else ""
                params_ = f"{args_}{sep_}{kwargs_}"
                print(f"{GREEN}{NAME}{END}({params_}) called.  {BLUE}(Call #{wrap.calls}){END}")
                start = time.time()

            # Call the original function:
            result = func(*args, **kwargs)

            if debug:
                end = time.time()
                duration = (end - start) * 10 ** 3
                # Log the return value:
                if isinstance(result, np.ndarray):
                    res_ = f"{VIOLET}Array{result.shape}{END}"
                elif isinstance(result, tuple):
                    res_ = ", ".join([f"{VIOLET}Array{a.shape}{END}" if isinstance(a, np.ndarray) else f"{YELLOW}{a}{END}" for a in result])
                else:
                    res_ = f"{YELLOW}{result}{END}"

                print(f"{RED}{NAME}{END}({params_}) returns {res_} after {duration:.3f} ms.")

            # Return the result:
            return result

        wrap.calls = 0
        return wrap

    return parameterized
