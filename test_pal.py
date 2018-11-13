from unittest import TestCase

from palindrome import pal


class TestPal(TestCase):
    def test_is_pal(self):
        self.assertTrue(pal(1001))

    def test_is_not_pal(self):
        self.assertFalse(pal(1002))

