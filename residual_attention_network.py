import tensorflow as tf
from tensorflow.contrib import slim

import arg_scope_
import attention_block
from residual_block import ResidualBlock


class ResidualAttentionNetwork(object):
    def __init__(self, is_training, num_class):
        self.is_training = is_training
        self.num_class = num_class
        self.residual_block = ResidualBlock(self.is_training)
        self.attention_block_stage0 = attention_block.AttentionBlockStage0(self.is_training)
        self.attention_block_stage1 = attention_block.AttentionBlockStage1(self.is_training)
        self.attention_block_stage2 = attention_block.AttentionBlockStage2(self.is_training)

    # 640*480
    def interface(self, input_x):
        with tf.variable_scope('residual_attention_network'):
            # resnet 头部结构，7*7，stride=2， 然后接一个2*2,stride=3的maxpool
            sc = arg_scope_.arg_scope(is_training=self.is_training)
            with slim.arg_scope(sc):
                conv1 = slim.conv2d(input_x, 64, [7, 7], stride=2, padding='SAME', scope='conv')
                mpool1 = slim.max_pool2d(conv1, [3, 3], stride=2, padding='SAME', scope='maxpool')

            residual_out1 = self.residual_block.residual_block(mpool1, 64, scope_name='residual_block1')
            # 缩小为1/8->80*60
            residual_out2 = self.residual_block.residual_block(
                residual_out1, 128, stride=2, scope_name='residual_block2'
            )
            # attention_stage1
            attention_out1 = self.attention_block_stage0.attention_block_stage0(residual_out2, 128, 1)

            # decode attention_out0
            # 上采样 变成1/2
            with slim.arg_scope(arg_scope_.arg_scope(is_training=self.is_training)):
                decode_attention_out1 = slim.conv2d(attention_out1, 128, [1, 1], stride=1, scope='deconv1-1')
                decode_attention_out1 = slim.conv2d_transpose(
                    decode_attention_out1, 64, [3, 3], stride=2, scope='deconv1-2'
                )
                decode_attention_out1 = slim.conv2d(decode_attention_out1, 64, [1, 1], stride=1, scope='deconv1-3')
                decode_attention_out1 = slim.conv2d_transpose(
                    decode_attention_out1, 1, [3, 3], stride=2,
                    normalizer_fn=None, activation_fn=None, scope='deconv1-4'
                )

            # 进行一步下采样
            # 缩小为1/16->40*30
            residual_out3 = self.residual_block.residual_block(
                attention_out1, 256, stride=2, scope_name='residual_block3'
            )
            # attention_stage1
            # attention_out1_1 = self.attention_block_stage1.attention_block_stage1(residual_out1, 256, 1)
            attention_out2_2 = self.attention_block_stage1.attention_block_stage1(residual_out3, 256, 2)

            # decode attention_out2
            # 上采样 变成1/4=
            with slim.arg_scope(arg_scope_.arg_scope(is_training=self.is_training)):
                decode_attention_out2 = slim.conv2d(attention_out2_2, 256, [1, 1], stride=1, scope='deconv2-1')
                decode_attention_out2 = slim.conv2d_transpose(
                    decode_attention_out2, 128, [3, 3], stride=2, scope='deconv2-2'
                )
                decode_attention_out2 = slim.conv2d(decode_attention_out2, 128, [1, 1], stride=1, scope='deconv2-3')
                decode_attention_out2 = slim.conv2d_transpose(
                    decode_attention_out2, 1, [3, 3], stride=2,
                    normalizer_fn=None, activation_fn=None, scope='deconv2-4'
                )

            # # 进行一步下采样
            # residual_out2 = self.residual_block.residual_block(
            #     attention_out1_2, 512, stride=2, scope_name='residual_block3'
            # )
            # # attention_stage2
            # # attention_out2_1 = self.attention_block_stage2.attention_block_stage2(residual_out2, 512, 1)
            # # attention_out2_2 = self.attention_block_stage2.attention_block_stage2(attention_out2_1, 512, 2)
            # attention_out2_3 = self.attention_block_stage2.attention_block_stage2(residual_out2, 512, 3)
            #
            # # decode attention_out2
            # with slim.arg_scope(arg_scope_.arg_scope(is_training=self.is_training)):
            #     decode_attention_out2 = slim.conv2d_transpose(
            #         attention_out2_3, 64, [3, 3], stride=2, scope='deconv3-1'
            #     )
            #     decode_attention_out2 = slim.conv2d_transpose(
            #         decode_attention_out2, 64, [3, 3], stride=2, scope='deconv3-2'
            #     )
            #     decode_attention_out2 = slim.conv2d_transpose(
            #         decode_attention_out2, 64, [3, 3], stride=2, scope='deconv3-3'
            #     )
            #     decode_attention_out2 = slim.conv2d_transpose(
            #         decode_attention_out2, 1, [3, 3], stride=2,
            #         normalizer_fn=None, activation_fn=None, scope='deconv3-4'
            #     )

            # 30*23
            # 20*15
            residual_out4 = self.residual_block.residual_block(
                attention_out2_2, 512, stride=2, scope_name='residual_block4'
            )
            residual_out5 = self.residual_block.residual_block(residual_out4, 512, scope_name='residual_block5')
            # 10*8
            residual_out6 = self.residual_block.residual_block(
                residual_out5, 1024, stride=2, scope_name='residual_block6'
            )
            global_avg_out = tf.reduce_mean(residual_out6, [1, 2], name='global_avg_pool', keepdims=True)
            logits = slim.conv2d(
                global_avg_out, self.num_class, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits'
            )

            return decode_attention_out1, decode_attention_out2, logits


if __name__ == '__main__':
    residual_attention_network = ResidualAttentionNetwork(is_training=True, num_class=2)
    input_x = tf.placeholder(tf.float32, [32, 480, 640, 3])
    decode_attention_out1, decode_attention_out2, logits = residual_attention_network.interface(input_x)
    print('attention_out1.shape:', decode_attention_out1.get_shape())
    print('attention_out2.shape:', decode_attention_out2.get_shape())
