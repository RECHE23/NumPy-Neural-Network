import time
import numpy as np

# Activates the trace for debugging:
SHOW_TRACE = False


def trace(display=SHOW_TRACE):
    # Colours for the trace:
    RED = '\33[31m'
    GREEN = '\33[32m'
    YELLOW = '\33[33m'
    BLUE = '\33[34m'
    VIOLET = '\33[35m'
    BEIGE = '\33[36m'
    END = "\x1b[0m"

    def parameterized(func):
        def wrap(*args, **kwargs):
            NAME = func.__qualname__ if func.__name__ != func.__qualname__ else f"{func.__module__}.{func.__name__}"
            if display:
                # Log the function name and arguments:
                argsname_ = func.__code__.co_varnames
                if argsname_[0] == 'self':
                    NAME = f"{args[0].__class__.__name__}.{func.__name__}"
                if NAME not in wrap.calls:
                    wrap.calls[NAME] = 0
                wrap.calls[NAME] += 1
                args_ = ", ".join([f"{k}={VIOLET}Array{v.shape}{END}" if isinstance(v, np.ndarray) else f"{k}={YELLOW}{v}{END}" \
                                   for k, v in zip(argsname_, args) if k != 'self'])
                kwargs_ = ", ".join([f"{k}={VIOLET}Array{v.shape}{END}" if isinstance(v, np.ndarray) else f"{k}={YELLOW}{v}{END}" \
                                     for k, v in kwargs.items()])
                sep_ = ", " if args_ and kwargs_ else ""
                params_ = f"{args_}{sep_}{kwargs_}"
                print(f"{GREEN}{NAME}{END}({params_}) called.  {BLUE}(Call #{wrap.calls[NAME]}){END}")
                start = time.time()

            # Call the original function:
            result = func(*args, **kwargs)

            if display:
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

        wrap.calls = dict()
        return wrap

    return parameterized


def debug_assert(condition, message):
    if __debug__:
        assert condition, message
