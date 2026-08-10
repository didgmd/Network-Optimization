import os.path
from inspect import getframeinfo, stack
from datetime import datetime

# 初始化调试模式
DEBUG_MODE = False


def set_debug_mode(mode):
    global DEBUG_MODE
    DEBUG_MODE = mode


def debug(msg):
    if not DEBUG_MODE:
        return

    caller = getframeinfo(stack()[1][0])
    filename = os.path.basename(caller.filename)
    print(f"{filename}:{caller.lineno} {msg}")


def debug_print(msg):
    caller = getframeinfo(stack()[1][0])
    filename = os.path.basename(caller.filename)
    print(
        f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} {filename}:{caller.lineno} {msg}"
    )
