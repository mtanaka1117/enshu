import tensorflow as tf


def apply_noise(inputs: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Applies a small amount of zero-mean noise to the give tensor.
    """
    noise = tf.random.uniform(shape=tf.shape(inputs),
                              minval=-1 * scale,
                              maxval=scale)
    return tf.add(inputs, noise)

