import importlib.util


def is_alibi_detect_available():
    return importlib.util.find_spec("alibi_detect") is not None
