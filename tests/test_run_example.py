import unittest
from ticket_matching import run_example


class RunExampleSmokeTest(unittest.TestCase):
    def test_run_example_returns_valid_structure(self):
        result = run_example()
        self.assertIsInstance(result, dict)
        self.assertIn("选中票据", result)
        self.assertIn("总金额", result)
        self.assertIn("综合得分", result)
        self.assertGreaterEqual(result["总金额"], 0)
        self.assertGreater(result["综合得分"], 0.0)


if __name__ == "__main__":
    unittest.main()
