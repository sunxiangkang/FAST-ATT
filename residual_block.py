import tensorflow as tf
from tensorflow.contrib import slim

import arg_scope_


class ResidualBlock:
    def __init__(self, is_training=True):
        """
        no params
        """
        self.is_training = is_training

    def batch_norm(self, input_x, is_training=True, scope_name=None):
        sc = arg_scope_.arg_scope(is_training=is_training)
        with tf.variable_scope(scope_name):
            with slim.arg_scope(sc):
                out = slim.batch_norm(input_x, scope_name='batch_norm')

                return out

    def residual_block(self, input_x, output_channels=None, stride=1, scope_name=None):
        """
        residual_block结构：
        1、 batch_norm+relu
        2、 1*1 卷积+batch_norm+relu
        3、 3*3卷积，stride采样+batch_norm+relu
        4、 1*1卷积
            如果输入通道数和输出通道数相等且residual_block的stride=1，将4的输出结果与1输出结果相加
            如果输入通道数和输出通道数不相等，或者residual_block的stride不等于1：执行5
        5、 1输出的结果进行1*1卷积，stride采样
            将5的输出结果和4的输出结果相加
        """
        input_channels = input_x.get_shape()[-1].value
        if output_channels is None:
            output_channels = input_channels
        scope = arg_scope_.arg_scope(is_training=self.is_training)
        with tf.variable_scope(scope_name):
            with slim.arg_scope(scope):
                x_1 = slim.batch_norm(input_x, scope='pre_batch_norm')
                x_ = slim.conv2d(x_1, output_channels//4, [1, 1], stride=1, padding='SAME', scope='conv1')
                x_ = slim.conv2d(x_, output_channels//4, [3, 3], stride=stride, padding='SAME', scope='conv2')
                x_ = slim.conv2d(
                    x_, output_channels, [1, 1], stride=1, padding='SAME',
                    normalizer_fn=None, activation_fn=None, scope='conv3'
                )

                if (input_channels != output_channels) or (stride != 1):
                    input_x = slim.conv2d(
                        x_1, output_channels, [1, 1], stride=stride, padding='SAME',
                        normalizer_fn=None, activation_fn=None, scope='conv_ad'
                    )
                output = x_ + input_x

                return output


if __name__ == '__main__':
    input_x = tf.placeholder(tf.float32, [32, 512, 512, 3])
    residual_block = ResidualBlock(True)
    out = residual_block.residual_block(input_x, 128, scope_name='residual_block')
    print(out.get_shape())
