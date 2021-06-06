import tensorflow as tf


def apply_noise(inputs: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Applies a small amount of zero-mean noise to the give tensor.
    """
    noise = tf.random.uniform(shape=tf.shape(inputs),
                              minval=-1 * scale,
                              maxval=scale)
    return tf.add(inputs, noise)


def interpolate_predictions(skip_predictions: tf.Tensor, update_gates: tf.Tensor) -> tf.Tensor:
    """
    Gets the collected indices from the given Skip RNN update gates.

    Args:
        skip_predictions: A [T, D] tensor of output features from the Skip RNN.
        update_gates: A [T] binary tensor (1 when collected, 0 when skipped)
    Returns:
        A [T, D] array of linearly-interpolated predictions
    """
    seq_length = tf.shape(update_gates)[0]

    update_gates = tf.cast(update_gates, dtype=tf.int64)
    seq_indices = tf.range(start=0, limit=seq_length, dtype=tf.int64)  # [T]
    collected_indices = tf.multiply(update_gates, seq_indices)  # [T]
    
    cumulative_gates = tf.cumsum(update_gates, axis=0) - 1  # [T]
    collected_mask = update_gates - 1  # [T]

    segment_ids = tf.multiply(cumulative_gates, update_gates) + collected_mask  # [T]
    num_segments = tf.reduce_sum(update_gates)

    # Remove the non-collected indices, [K]
    cleaned_indices = tf.math.unsorted_segment_sum(data=collected_indices,
                                                   segment_ids=segment_ids,
                                                   num_segments=num_segments)

    # Fetch the corresponding measurements, [K, D]
    collected_measurements = tf.math.unsorted_segment_sum(data=skip_predictions,
                                                          segment_ids=segment_ids,
                                                          num_segments=num_segments)

    last_diff = tf.cast(seq_length, cleaned_indices.dtype) - cleaned_indices[-1]
    diffs = tf.concat([cleaned_indices[1:] - cleaned_indices[:-1], [last_diff]], axis=0)  # [K]

    slopes = tf.roll(collected_measurements, shift=-1, axis=0) - collected_measurements  # [K, D] (last element on axis 0 is garbage)
    slopes = slopes / tf.expand_dims(tf.cast(diffs, dtype=slopes.dtype), axis=-1)

    num_indices = tf.shape(diffs)[0]
    slope_mask = tf.cast(tf.range(start=0, limit=num_indices) < (num_indices - 1), dtype=slopes.dtype)  # [K]
    slopes = tf.multiply(slopes, tf.expand_dims(slope_mask, axis=-1))

    repeated_indices = tf.repeat(cleaned_indices, repeats=diffs)  # [T]
    repeated_slopes = tf.repeat(slopes, repeats=diffs, axis=0)  # [T, D]
    repeated_intercepts = tf.repeat(collected_measurements, repeats=diffs, axis=0)  # [T, D]

    slope_indices = tf.cast(seq_indices - repeated_indices, dtype=repeated_slopes.dtype)  # [T]
    slope_indices = tf.expand_dims(slope_indices, axis=1)    

    interpolated = (repeated_slopes * slope_indices) + repeated_intercepts

    return interpolated


def batch_interpolate_predictions(skip_predictions: tf.Tensor, update_gates: tf.Tensor) -> tf.Tensor:
    """
    Performs linear interpolation on the batch of collected measurements.

    Args:
        skip_predictions: A [B, T, D] tensor of output features
        update_gates: A [B, T] tensor of binary update gates
    Returns:
        A [B, T, D] tensor of linearly interpolated predictions.
    """
    batch_size = tf.shape(skip_predictions)[0] 

    # Tensor array to store result values for each batch sample
    result_array = tf.TensorArray(dtype=skip_predictions.dtype,
                                  size=batch_size,
                                  dynamic_size=False,
                                  clear_after_read=True)

    def cond(index: tf.Tensor, _):
        return index < batch_size

    def body(index: tf.Tensor, results: tf.TensorArray):
        skip_pred = tf.gather(skip_predictions, indices=index, axis=0)
        gates = tf.gather(update_gates, indices=index, axis=0)

        interpolated = interpolate_predictions(skip_pred, gates)  # [T, D]

        results = results.write(index=index, value=interpolated)

        return [index + 1, results]
    
    index = tf.constant(0)
    _, results = tf.while_loop(cond=cond,
                               body=body,
                               loop_vars=[index, result_array],
                               maximum_iterations=batch_size)

    return results.stack()



#input_array = [[[8], [2], [2], [6], [6], [7]], [[5], [5], [4], [4], [4], [4]]]
#update_gates = [[1, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 0]]
#
#with tf.compat.v1.Session(graph=tf.Graph()) as sess:
#
#
#    input_ph = tf.compat.v1.placeholder(shape=(None, 6, 1),
#                                        dtype=tf.float32,
#                                        name='inputs')
#
#    gates_ph = tf.compat.v1.placeholder(shape=(None, 6),
#                                        dtype=tf.float32,
#                                        name='gates')
#
#
#    #skip_predictions = tf.constant([[8, 0], [2, 2], [2, 2], [6, 6], [6, 6], [7, 7]], dtype=tf.float32)
#    #skip_predictions = tf.compat.v1.get_variable(shape=(6, 2),
#    #                                             initializer=tf.compat.v1.random_normal_initializer(),
#    #                                             dtype=tf.float32,
#    #                                             name='skip',
#    #                                             trainable=True)
#    
#    #update_gates = tf.constant([1, 1, 0, 1, 0, 0], dtype=tf.float32)
#
#    output = batch_interpolate_predictions(input_ph, gates_ph)
#    #output = interpolate_predictions(skip_predictions, update_gates)
#
#    sess.run(tf.compat.v1.global_variables_initializer())
#    result = sess.run(output, feed_dict={input_ph: input_array, gates_ph: update_gates})
#
#    print(result)
#
