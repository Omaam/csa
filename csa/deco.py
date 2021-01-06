import time
import os


def stopwatch(func):
    def wrapper(*arg, **kargs):
        start = time.time()
        print(f'start {func.__name__}')
        res = func(*arg, **kargs)
        dura = (time.time() - start)
        print(time.strftime(f'finish {func.__name__}: %H:%M\'%S\"',
                            time.gmtime(dura)))
        return res
    return wrapper


def change_directory(path_to_dir):
    def _change_directory(func):
        def wrapper(*args, **kargs):
            current_dir = os.getcwd()
            if os.path.exists(path_to_dir) is False:
                os.makedirs(path_to_dir)
            os.chdir(path_to_dir)
            results = func(*args, **kargs)
            os.chdir(current_dir)
            return(results)
        return wrapper
    return _change_directory
