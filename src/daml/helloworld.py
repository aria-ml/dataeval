class HelloWorld:
    def __init__(self, message: str = "Hello World :p") -> None:
        self.message = message

    def run(self):
        print(self.message)
        return self.message
