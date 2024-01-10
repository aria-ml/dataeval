import importlib.util


def is_alibi_detect_available():
    return (
        importlib.util.find_spec("tensorflow") is not None
        and importlib.util.find_spec("tensorflow_probability") is not None
    )


def is_maite_available():
    return importlib.util.find_spec("maite") is not None
