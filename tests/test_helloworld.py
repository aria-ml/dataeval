import unittest

from daml.helloworld import HelloWorld


class TestHelloWorld(unittest.TestCase):
    def test_hello_world(self):
        h_world = HelloWorld()
        self.assertEqual(h_world.run(), h_world.message)
