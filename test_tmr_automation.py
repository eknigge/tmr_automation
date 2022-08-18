from tmr_automation import *
from unittest import TestCase


class TestAutomationTMR(TestCase):

    def test_expand_transmittal_headers(self):
        test_transmittal_headers = [('HEADER1', 3), ('HEADER2', 8)]
        output = expand_transmittal_headers(test_transmittal_headers)
        self.assertEqual(len(output),
                         headers[0][1] + headers[1][1])
        self.assertEqual(output[0], 'HEADER1_')

    def test_combine_headers(self):
        list1 = ['1', '2', '3']
        list2 = ['2', '3', '4']
        list3 = ['a', 'b']
        expected = ['12', '23', '34']
        output = combine_headers(list1, list2)
        self.assertEqual(expected, output)
        with self.assertRaises(ValueError):
            combine_headers(list1, list3)
