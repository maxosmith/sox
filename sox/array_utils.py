import numpy as np


def stack_ragged_arrays(arrays, max_len: int | None = None, padding_value: int = 0):
  """Stacks a list of ragged arrays (arrays of different lengths) into a single numpy array.

  Args:
    arrays: The list of arrays to stack.
    max_len: The maximum length of the arrays in the stacked array.
                                If not specified, it defaults to the length of the longest array.
    padding_value: The value to use for padding shorter arrays. Defaults to 0.

  Returns:
    numpy.ndarray: A stacked array of the input arrays.
  """
  if not arrays:  # Check if the array list is empty
    return np.array([], dtype=int)

  # Find the maximum length if not provided
  if max_len is None:
    max_len = max(len(arr) for arr in arrays)

  # Initialize the stacked array with padding values
  stacked_array = np.full((len(arrays), max_len), padding_value, dtype=arrays[0].dtype)

  # Assign the values from the original arrays to the stacked array
  for i, arr in enumerate(arrays):
    stacked_array[i, : len(arr)] = arr

  return stacked_array


def unstack(a: np.ndarray, axis: int = 0):
  """Unstack an axis of an array.

  Args:
    a: Array to unstack.
    axis: Axis to remove.

  Returns:
    List of arrays that are from `a`.
  """
  return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis=axis)]
