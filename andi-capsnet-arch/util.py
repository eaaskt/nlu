import numpy as np
import tensorflow as tf


def norm_matrix(matrix):
    """
        normalize matrix by column
        input : numpy array, dtype = float32
        output : normalized numpy array, dtype = float32
      """
    # check dtype of the input matrix
    np.testing.assert_equal(type(matrix).__name__, 'ndarray')
    np.testing.assert_equal(matrix.dtype, np.float32)

    row_sums = matrix.sum(axis=1)
    # replace zero denominator
    row_sums[row_sums == 0] = 1
    norm_matrix = matrix / row_sums[:, np.newaxis]
    return norm_matrix


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name='safe_norm'):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

