class _VerboseSingleton:
    verbose = False

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(_VerboseSingleton, cls).__new__(cls)
        return cls.instance


def verbose(text: str):
    if _VerboseSingleton().verbose:
        print(text)


def set_verbose(value: bool):
    _VerboseSingleton().verbose = value
