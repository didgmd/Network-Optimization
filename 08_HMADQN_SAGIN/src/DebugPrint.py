import os.path
from inspect import getframeinfo, stack
from datetime import datetime

DEBUG_MODE = False


def set_debug_mode(mode: bool):
    global DEBUG_MODE
    DEBUG_MODE = bool(mode)


def debug(msg: str):
    if not DEBUG_MODE:
        return
    caller = getframeinfo(stack()[1][0])
    filename = os.path.basename(caller.filename)
    print(f"{filename}:{caller.lineno} {msg}")


def debug_print(msg: str):
    caller = getframeinfo(stack()[1][0])
    filename = os.path.basename(caller.filename)
    print(
        f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} {filename}:{caller.lineno} {msg}"
    )
