import tensorflow as tf
from tensorflow.contrib import slim


def arg_scope(
        is_training=True, weight_decay=1e-4, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True
):
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': None
    }
    with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.xavier_initializer(),
            activation_fn=slim.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params
    ):
        with slim.arg_scope(
                [slim.batch_norm],
                **batch_norm_params,
                activation_fn=tf.nn.relu
        ) as sc:
            return sc
