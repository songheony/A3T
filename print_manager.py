import os
import sys
from contextlib import contextmanager, redirect_stdout


@contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    code from https://stackoverflow.com/a/17753573/9811770
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, "w")
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


def do_not_print(func):
    def wrapper(*args, **kwargs):
        with stdchannel_redirected(sys.stderr, os.devnull), redirect_stdout(
            open(os.devnull, "w")
        ):
            return func(*args, **kwargs)

    return wrapper
