import numpy as np

# Activates the trace for debugging:
DEBUG = False


def convert_targets(targets, to=None):
    if (to is None and targets.ndim == 1) or to == "categorical":
        # Equivalent to keras.utils.to_categorical:
        targets = np.eye(len(set(targets)))[targets]
    elif to == "one_hot":
        # Converts targets' vectors to one hot vectors:
        idx = np.argmax(targets, axis=-1)
        targets = np.zeros(targets.shape)
        targets[np.arange(targets.shape[0]), idx] = 1
    elif to == "binary":
        # Converts a probability vector to a binary vector using a threshold of 0.5:
        targets = np.where(targets >= 0.5, 1, 0)
    elif to == "labels" and targets.ndim != 1:
        # Converts targets' vectors to labels:
        targets = np.argmax(targets, axis=-1)
    elif to == "probability":
        # Converts targets' vectors to probability distributions:
        from activation_functions import softmax
        targets = softmax(targets)
    return targets


def batch_iterator(samples, targets, batch_size, shuffle=False):
    assert samples.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(samples.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, samples.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, samples.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield samples[excerpt], targets[excerpt]


def accuracy_score(y_true, y_predicted):
    y_true = convert_targets(y_true, to="labels")
    y_predicted = convert_targets(y_predicted, to="labels")
    return (y_true == y_predicted).sum() / y_true.shape[0]


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
            if debug:
                # Log the function name and arguments:
                argsname_ = func.__code__.co_varnames
                args_ = ", ".join([f"{k}={VIOLET}Array{v.shape}{END}" if isinstance(v, np.ndarray) else f"{k}={YELLOW}{v}{END}" \
                                   for k, v in zip(argsname_, args) if k != 'self'])
                kwargs_ = ", ".join([f"{k}={VIOLET}Array{v.shape}{END}" if isinstance(v, np.ndarray) else f"{k}={YELLOW}{v}{END}" \
                                     for k, v in kwargs.items()])
                sep_ = ", " if args_ and kwargs_ else ""
                params_ = f"{args_}{sep_}{kwargs_}"
                print(f"{GREEN}{NAME}{END}({params_}) called.")

            # Call the original function:
            result = func(*args, **kwargs)

            if debug:
                # Log the return value:
                if isinstance(result, np.ndarray):
                    res_ = f"{VIOLET}Array{result.shape}{END}"
                elif isinstance(result, tuple):
                    res_ = ", ".join([f"{VIOLET}Array{a.shape}{END}" if isinstance(a, np.ndarray) else f"{YELLOW}{a}{END}" for a in result])
                else:
                    res_ = f"{YELLOW}{result}{END}"

                print(f"{RED}{NAME}{END}({params_}) returns {res_}.")

            # Return the result:
            return result

        return wrap

    return parameterized
