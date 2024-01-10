import importlib.util


def is_maite_available():
    return importlib.util.find_spec("maite") is not None
