import sys


class HelloWorld:
    def __init__(self, message: str = "Hello World :p") -> None:
        self.message = message

    def run(self) -> str:
        result: str
        if sys.version_info[:2] == (3, 10):
            result = self.message
        elif sys.version_info[:2] == (3, 9):
            result = self.message
        elif sys.version_info[:2] == (3, 8):
            result = self.message
        else:  # pragma: no cover
            result = "Unsupported Python"
        print(result)
        return result
