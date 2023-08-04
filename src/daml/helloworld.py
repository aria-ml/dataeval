import sys


class HelloWorld:
    """
    An object containing the message as provided.
    If no message is provided, then woe, 'Hello World :p' be upon ye.

    :param message: Optional message to store, or 'Hello World :p' by default.
    :type message: str

    """

    def __init__(self, message: str = "Hello World :p") -> None:
        self.message = message

    def run(self) -> str:
        """
        Prints and returns the message in HelloWorld.

        :return: The message in HelloWorld.
        :rtype: str

        """
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
