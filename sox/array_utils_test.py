"""Test for `stack_ragged_arrays` function."""
import numpy as np
from absl.testing import absltest, parameterized

from sox import array_utils


class StackRaggedArraysTest(parameterized.TestCase):
    """Test suite for `stack_ragged_arrays`."""

    @parameterized.parameters(
        # Test case: regular arrays
        {"arrays": [[1, 2, 3], [4, 5], [6]], "expected": np.array([[1, 2, 3], [4, 5, 0], [6, 0, 0]])},
        # Test case: empty list
        {"arrays": [], "expected": np.array([], dtype=int)},
        # Test case: arrays of same length
        {"arrays": [[1, 2], [3, 4]], "expected": np.array([[1, 2], [3, 4]])},
        # Test case: specified max_length and padding_value
        {
            "arrays": [[1], [2], [3]],
            "max_length": 5,
            "padding_value": -1,
            "expected": np.array([[1, -1, -1, -1, -1], [2, -1, -1, -1, -1], [3, -1, -1, -1, -1]]),
        },
    )
    def test_stack_ragged_arrays(self, arrays, expected, max_length=None, padding_value=0):
        """Tests `stack_ragged_arrays` with various parameters."""
        result = array_utils.stack_ragged_arrays([np.asarray(a) for a in arrays], max_length, padding_value)
        np.testing.assert_array_equal(result, expected)

    @parameterized.parameters(
        {
            "array": [[[2, 1, 0], [0, -1, -2]], [[2, 1, 0], [0, -1, -2]]],
            "axis": 0,
        },
        {
            "array": [[[2, 1, 0], [0, -1, -2]], [[2, 1, 0], [0, -1, -2]]],
            "axis": 1,
        },
    )
    def test_unstack(self, array, axis):
        """Test `unstack`."""
        stacked = np.stack([np.asarray(a) for a in array], axis=axis)
        unstacked = array_utils.unstack(stacked, axis=axis)
        for expected, actual in zip(unstacked, array):
            np.testing.assert_array_almost_equal(expected, np.asarray(actual))

    @parameterized.parameters(
        {
            "array": [[1, 2, 3], [4, 5, 6]],
            "expected_shape": (2, 3),
            "expected_value": 1.0 / 6,
        },
        {
            "array": [[1], [2], [3]],
            "expected_shape": (3, 1),
            "expected_value": 1.0 / 3,
        },
        {
            "array": [1, 2, 3, 4, 5],
            "expected_shape": (5,),
            "expected_value": 1.0 / 5,
        },
        {
            "array": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            "expected_shape": (2, 2, 2),
            "expected_value": 1.0 / 8,
        },
        {
            "array": [],
            "expected_shape": (0,),
            "expected_value": 1.0,
        },
    )
    def test_uniform_like(self, array, expected_shape, expected_value):
        """Tests `uniform_like` with various array shapes."""
        result = array_utils.uniform_like(np.asarray(array))
        self.assertEqual(result.shape, expected_shape)
        if expected_shape != (0,):
            np.testing.assert_allclose(result, np.full(expected_shape, expected_value))
        else:
            self.assertTrue(np.isnan(result).all())  # Handle empty array case.


if __name__ == "__main__":
    absltest.main()
