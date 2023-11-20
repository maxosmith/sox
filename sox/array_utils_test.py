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


if __name__ == "__main__":
  absltest.main()
