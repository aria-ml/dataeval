import unittest

from daml.helloworld import HelloWorld


class TestHelloWorld(unittest.TestCase):
    def test_hello_world(self):
        test_message = "Hello World! :p"
        h_world = HelloWorld(message=test_message)
        self.assertEqual(h_world.run(), test_message)
