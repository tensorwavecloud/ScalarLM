def is_main_rank():
    return True


def barrier():
    pass


def main_rank_only(func):
    def wrap_function(*args, **kwargs):
        result = None
        barrier()
        if is_main_rank():
            result = func(*args, **kwargs)
        barrier()
        return result

    return wrap_function
