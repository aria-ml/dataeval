class _VerboseSingleton:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(_VerboseSingleton, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.verbose = False


v = _VerboseSingleton()


def verbose(text: str):
    if v.instance.verbose:
        print(text)


def set_verbose(value: bool):
    v.instance.verbose = value
