"""Regression checks: python -m unittest test_interface -v."""
import unittest
from interface import validate, DEFAULTS


class ParameterTests(unittest.TestCase):
    def test_browser_integer_weights_work_in_backward_scan(self):
        # JSON number fields such as 2000 arrive as integers.
        from pendubot import Config, Solver
        settings = validate({'qn0':2000, 'qn1':2000, 'qn2':200, 'qn3':200})
        solver = Solver(Config(duration=1.,dt=.05,q_terminal=tuple(settings['qn'+str(i)] for i in range(4))))
        solver.warmup()  # Previously failed with incompatible scan carry dtypes.

    def test_defaults_are_valid(self):
        self.assertEqual(validate(DEFAULTS), DEFAULTS)

    def test_timing_must_align(self):
        with self.assertRaisesRegex(ValueError,'exact multiple'):
            validate({'duration':6.01})

    def test_kick_must_precede_arrival(self):
        with self.assertRaisesRegex(ValueError,'before the end'):
            validate({'kick1':.1,'kick_time':6.})

    def test_nonfinite_and_unknown_parameters_rejected(self):
        for settings in ({'m1':float('nan')},{'m2':-1.},{'extra':1},{'r':True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate(settings)

    def test_integer_iteration_budget(self):
        with self.assertRaisesRegex(ValueError,'whole number'):
            validate({'max_iterations':1500.5})


if __name__ == '__main__':
    unittest.main()
