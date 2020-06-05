import tensorflow as tf
from tensorflow.contrib import slim

import arg_scope_
from residual_block import ResidualBlock


class AttentionBlockStage0(object):
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.residual_block = ResidualBlock(self.is_training)

    def attention_block_stage0(self, input_x, output_channels, sub_scope_num):
        with tf.variable_scope('attention_block_stage0-{}'.format(sub_scope_num)):
            # 1、residual_attention_bolck 之前的卷积
            x = self.residual_block.residual_block(input_x, output_channels, scope_name='head_block')

            # 2、输出x分为两个分支
            # 2.1分支：2个residual_block堆叠，输出trunk
            out_trunk = self.residual_block.residual_block(x, output_channels, scope_name='trunk_block1')
            out_trunk = self.residual_block.residual_block(out_trunk, output_channels, scope_name='trunk_block2')
            # 2.2 分支：attention分支
            #   先是一个max_pool+一个残差block
            out_mpool1 = slim.max_pool2d(x, [3, 3], stride=2, padding='SAME', scope='max_pool1')
            out_softmax1 = self.residual_block.residual_block(out_mpool1, output_channels, scope_name='out_softmax1')

            # 3、out_softmax1又分为两个分支
            # 3.1 skip_connection分支
            out_skip_connection1 = self.residual_block.residual_block(
                out_softmax1, output_channels, scope_name='out_skip_connection1'
            )
            # 3.2 相当于是有一个attentin分支:一个max_pool+一个残差block
            out_mpool2 = slim.max_pool2d(out_softmax1, [3, 3], stride=2, padding='SAME', scope='max_pool2')
            out_softmax2 = self.residual_block.residual_block(
                out_mpool2, output_channels, scope_name='out_softmax2'
            )

            # 4、out_softmax2又是两个分支
            # 3.1 skip_connection分支
            out_skip_connection2 = self.residual_block.residual_block(
                out_softmax2, output_channels, scope_name='out_skip_connection2'
            )
            # 3.2 相当于是一个attention分支：一个max_pool+一个残差block
            out_mpool3 = slim.max_pool2d(out_softmax2, [3, 3], stride=2, padding='SAME', scope='max_pool3')
            out_softmax3 = self.residual_block.residual_block(
                out_mpool3, output_channels, scope_name='out_softmax3'
            )

            # out_softmax3又分成两个分支
            # 5.1 skip_connection分支
            out_skip_connection3 = self.residual_block.residual_block(
                out_softmax3, output_channels, scope_name='out_skip_connection3'
            )
            # 5.2 相当于一个attention分支：一个max_pool+一个残差block
            out_pool4 = slim.max_pool2d(out_softmax3, output_channels, padding='SAME', scope='max_pool4')
            out_softmax4 = self.residual_block.residual_block(
                out_pool4, output_channels, scope_name='out_softmax4-1'
            )

            # -------------------分支分完了，下面进行合并-------------------

            # 将output_softmax4 做一个residual_block,然后做上采样，和out_softmax4,skip_connection3 相加
            # 6、 interpolation4
            out_softmax4 = self.residual_block.residual_block(
                out_softmax4, output_channels, scope_name='out_softmax4-2'
            )
            out_interp4 = tf.image.resize(
                out_softmax4,
                out_softmax3.get_shape()[1:3],
                tf.image.ResizeMethod.BILINEAR
            ) + out_softmax3
            out = out_interp4 + out_skip_connection3

            # 将out_interp4 做一个residual_block, 然后上采样，和和out_softmax3,skip_connection2 相加
            # 7、 interpolation3
            out_softmax5 = self.residual_block.residual_block(out, output_channels, scope_name='out_softmax5')
            out_interp3 = tf.image.resize(
                out_softmax5,
                out_softmax2.get_shape()[1:3],
                tf.image.ResizeMethod.BILINEAR
            ) + out_softmax2
            out = out_interp3 + out_skip_connection2

            # 将out_interp3 做一个residual_block，然后上采样，和out_softmax2,skip_connection1 相加
            # 8、interpolation2
            out_softmax6 = self.residual_block.residual_block(out, output_channels, scope_name='out_softmax6')
            out_interp2 = tf.image.resize(
                out_softmax6,
                out_softmax1.get_shape()[1:3],
                tf.image.ResizeMethod.BILINEAR
            ) + out_softmax1
            out = out_interp2 + out_skip_connection1

            # 将out_interp2 做一个residual_block，然后上采样，和out_softmax1,x_trunk 相加
            # 9、 interpolation1
            out_softmax7 = self.residual_block.residual_block(out, output_channels, scope_name='out_softmax7')
            out_interp1 = tf.image.resize(
                out_softmax7,
                out_trunk.get_shape()[1:3],
                tf.image.ResizeMethod.BILINEAR
            ) + out_trunk

            # batch+relu+conv+batch+relu+conv+sigmoid
            # 10、 out_softmax8
            with tf.variable_scope('out_softmax8'):
                with slim.arg_scope(arg_scope_.arg_scope(is_training=self.is_training)):
                    out = slim.batch_norm(out_interp1, scope='batch_norm')
                    out = slim.conv2d(out, output_channels, [1, 1], stride=1, scope='conv1')
                    out = slim.conv2d(
                        out, output_channels, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv2'
                    )
                    out_softmax8 = tf.nn.sigmoid(out)

            # element_wise操作
            # 11、 attention
            out = (1+out_softmax8)*out_trunk
            # element_add操作
            # 12、 last_out
            out_last = self.residual_block.residual_block(out, output_channels, scope_name='last_out')

            return out_last


class AttentionBlockStage1(object):
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.residual_block = ResidualBlock(self.is_training)

    def attention_block_stage1(self, input_x, output_channels, sub_scope_num):
        with tf.variable_scope('attention_block_stage1-{}'.format(sub_scope_num)):
            # 1、residual_attention_bolck 之前的卷积
            x = self.residual_block.residual_block(input_x, output_channels, scope_name='head_block')

            # 2、输出x分为两个分支
            # 2.1分支：2个residual_block堆叠，输出trunk
            out_trunk = self.residual_block.residual_block(x, output_channels, scope_name='trunk_block1')
            out_trunk = self.residual_block.residual_block(out_trunk, output_channels, scope_name='trunk_block2')
            # 2.2 分支：attention分支
            #   先是一个max_pool+一个残差block
            out_mpool1 = slim.max_pool2d(x, [3, 3], stride=2, padding='SAME', scope='max_pool1')
            out_softmax1 = self.residual_block.residual_block(out_mpool1, output_channels, scope_name='out_softmax1')

            # 3、out_softmax1又分为两个分支
            # 3.1 skip_connection分支
            out_skip_connection1 = self.residual_block.residual_block(
                out_softmax1, output_channels, scope_name='out_skip_connection1'
            )
            # 3.2 相当于是有一个attentin分支:一个max_pool+一个残差block
            out_mpool2 = slim.max_pool2d(out_softmax1, [3, 3], stride=2, padding='SAME', scope='max_pool2')
            out_softmax2 = self.residual_block.residual_block(
                out_mpool2, output_channels, scope_name='out_softmax2'
            )

            # 4、out_softmax2又是两个分支
            # 4.1 skip_connection分支
            out_skip_connection2 = self.residual_block.residual_block(
                out_softmax2, output_channels, scope_name='out_skip_connection2'
            )
            # 4.2 相当于是一个attention分支：一个max_pool+一个残差block
            out_mpool3 = slim.max_pool2d(out_softmax2, [3, 3], stride=2, padding='SAME', scope='max_pool3')
            out_softmax3 = self.residual_block.residual_block(out_mpool3, output_channels, scope_name='out_softmax3-1')

            # -------------------分支分完了，下面进行合并-------------------

            # 将output_softmax3 做一个residual_block,然后做上采样，和out_softmax2,skip_connection2 相加
            # 5、 interpolation3
            out_softmax3 = self.residual_block.residual_block(
                out_softmax3, output_channels, scope_name='out_softmax3-2'
            )
            out_interp3 = tf.image.resize(
                out_softmax3,
                out_softmax2.get_shape()[1:3],
                tf.image.ResizeMethod.BILINEAR
            ) + out_softmax2
            out = out_interp3 + out_skip_connection2

            # 将out_interp3 做一个residual_block, 然后上采样，和和out_softmax1,skip_connection1 相加
            # 6、 interpolation2
            out_softmax4 = self.residual_block.residual_block(out, output_channels, scope_name='out_softmax4')
            out_interp2 = tf.image.resize(
                out_softmax4,
                out_softmax1.get_shape()[1:3],
                tf.image.ResizeMethod.BILINEAR
            ) + out_softmax1
            out = out_interp2 + out_skip_connection1

            # 将out_interp2 做一个residual_block，然后上采样，和x_trunk 相加
            # 7、 interpolation1
            out_softmax5 = self.residual_block.residual_block(out, output_channels, scope_name='out_softmax5')
            out_interp1 = tf.image.resize(
                out_softmax5,
                out_trunk.get_shape()[1:3],
                tf.image.ResizeMethod.BILINEAR
            ) + out_trunk

            # batch+relu+conv+batch+relu+conv+sigmoid
            # 8、 out_softmax6
            with tf.variable_scope('out_softmax6'):
                with slim.arg_scope(arg_scope_.arg_scope(is_training=self.is_training)):
                    out = slim.batch_norm(out_interp1, scope='batch_norm')
                    out = slim.conv2d(out, output_channels, [1, 1], stride=1, scope='conv1')
                    out = slim.conv2d(
                        out, output_channels, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv2'
                    )
                    out_softmax6 = tf.nn.sigmoid(out)

            # element_wise操作
            # 9、 attention
            out = (1 + out_softmax6) * out_trunk
            # element_add操作
            # 10、 last_out
            out_last = self.residual_block.residual_block(out, output_channels, scope_name='last_out')

            return out_last


class AttentionBlockStage2(object):
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.residual_block = ResidualBlock(self.is_training)

    def attention_block_stage2(self, input_x, output_channels, sub_scope_num):
        with tf.variable_scope('attention_block_stage2-{}'.format(sub_scope_num)):
            # 1、residual_attention_bolck 之前的卷积
            x = self.residual_block.residual_block(input_x, output_channels, scope_name='head_block')

            # 2、输出x分为两个分支
            # 2.1分支：2个residual_block堆叠，输出trunk
            out_trunk = self.residual_block.residual_block(x, output_channels, scope_name='trunk_block1')
            out_trunk = self.residual_block.residual_block(out_trunk, output_channels, scope_name='trunk_block2')
            # 2.2 分支：attention分支
            #   先是一个max_pool+一个残差block
            out_mpool1 = slim.max_pool2d(x, [3, 3], stride=2, padding='SAME', scope='max_pool1')
            out_softmax1 = self.residual_block.residual_block(out_mpool1, output_channels, scope_name='out_softmax1')

            # 3、out_softmax1又分为两个分支
            # 3.1 skip_connection分支
            out_skip_connection1 = self.residual_block.residual_block(
                out_softmax1, output_channels, scope_name='out_skip_connection1'
            )
            # 3.2 相当于是有一个attentin分支:一个max_pool+一个残差block
            out_mpool2 = slim.max_pool2d(out_softmax1, [3, 3], stride=2, padding='SAME', scope='max_pool2')
            out_softmax2 = self.residual_block.residual_block(
                out_mpool2, output_channels, scope_name='out_softmax2-1'
            )

            # -------------------分支分完了，下面进行合并-------------------

            # 将output_softmax2 做一个residual_block,然后做上采样，和out_softmax1,skip_connection1 相加
            # 4、 interpolation2
            out_softmax2 = self.residual_block.residual_block(
                out_softmax2, output_channels, scope_name='out_softmax2-2'
            )
            out_interp2 = tf.image.resize(
                out_softmax2,
                out_softmax1.get_shape()[1:3],
                tf.image.ResizeMethod.BILINEAR
            ) + out_softmax1
            out = out_interp2 + out_skip_connection1

            # 将out_interp2 做一个residual_block，然后上采样，和x_trunk 相加
            # 5、 interpolation1
            out_softmax3 = self.residual_block.residual_block(out, output_channels, scope_name='out_softmax3')
            out_interp1 = tf.image.resize(
                out_softmax3,
                out_trunk.get_shape()[1:3],
                tf.image.ResizeMethod.BILINEAR
            ) + out_trunk

            # batch+relu+conv+batch+relu+conv+sigmoid
            # 6、 out_softmax4
            with tf.variable_scope('out_softmax4'):
                with slim.arg_scope(arg_scope_.arg_scope(is_training=self.is_training)):
                    out = slim.batch_norm(out_interp1, scope='batch_norm')
                    out = slim.conv2d(out, output_channels, [1, 1], stride=1, scope='conv1')
                    out = slim.conv2d(
                        out, output_channels, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv2'
                    )
                    out_softmax6 = tf.nn.sigmoid(out)

            # element_wise操作
            # 7、 attention
            out = (1 + out_softmax6) * out_trunk
            # element_add操作
            # 8、 last_out
            out_last = self.residual_block.residual_block(out, output_channels, scope_name='last_out')

            return out_last


if __name__ == '__main__':
    input_x = tf.placeholder(tf.float32, [32, 512, 512, 3])
    output_channels = [128, 256, 512]
    attention_block_stage0 = AttentionBlockStage0(True)
    out1 = attention_block_stage0.attention_block_stage0(input_x, output_channels[0], 0)
    print(out1.get_shape())
    attention_block_stage1 = AttentionBlockStage1(True)
    out2 = attention_block_stage1.attention_block_stage1(out1, output_channels[1], 0)
    print(out2.get_shape())
    attention_block_stage2 = AttentionBlockStage2(True)
    out3 = attention_block_stage2.attention_block_stage2(out2, output_channels[2], 0)
    print(out3.get_shape())
