import unittest

from src.modules.module_core import module_core_cmd_help, module_core_cmd_exit


class test_module_core(unittest.TestCase):
    def test_module_core_help(self):
        module_core_cmd_help_instance = module_core_cmd_help()
        print(module_core_cmd_help_instance.short())
        print(module_core_cmd_help_instance.long())
