import unittest

from src.modules.module_core import module_core_cmd_help, module_core_cmd_exit
from src.launch import parse_arguments


def setUpModule():
    parse_arguments(["--modules", "core", "io"])


def tearDownModule():
    kernel.clean()


class test_module_core(unittest.TestCase):
    def test_module_core_help(self):
        module_core_cmd_help_instance = module_core_cmd_help()
        print(module_core_cmd_help_instance.short())
        print(module_core_cmd_help_instance.long())

    def test_module_core_exit(self):
        module_core_cmd_exit_instance = module_core_cmd_exit()
        print(module_core_cmd_exit_instance.short())
