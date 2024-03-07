import sys


class TestImports:
    def test_no_prototype_code(self):
        import daml

        assert daml is not None
        modules = sys.modules.keys()
        assert "daml._prototype" not in modules
