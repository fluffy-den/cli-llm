import unittest
import sys
import os

sys.path.append(os.path.abspath("./src"))  # Add the src directory to the Python path


def run_tests():
    # Define the path to the tests folder
    test_folder = "tests"

    # Discover all test cases in the tests folder
    loader = unittest.TestLoader()
    suite = loader.discover(test_folder)

    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    run_tests()
