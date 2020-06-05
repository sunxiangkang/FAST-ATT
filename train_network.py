import os
import cv2
import logging
import numpy as np
import tensorflow as tf
from multiprocessing import Pool

from residual_attention_network import ResidualAttentionNetwork

# 定义log
logger = logging.getLogger('train_resnet')
logger.setLevel(level=logging.INFO)
if not os.path.exists(r'./log'):
    os.makedirs(r'./log')
handler = logging.FileHandler('./log/logging.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler_con = logging.StreamHandler()
handler_con.setLevel(logging.INFO)
handler_con.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(handler_con)


def main(reader, batch_size=16, epoches=1000, input_shape=(480, 640), num_class=2, learning_rate=5e-4, reg_l2=False):
    input_x = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], 3])
    input_mask = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], 1])
    labels = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool)

    residual_attention_network = ResidualAttentionNetwork(is_training, num_class=num_class)
    decode_attention_out1, decode_attention_out2, logits = residual_attention_network.interface(input_x)
    att_mask1 = tf.cast(tf.where(
        tf.nn.sigmoid(decode_attention_out1) > 0.5,
        tf.ones_like(decode_attention_out1),
        tf.zeros_like(decode_attention_out1))*255, tf.uint8
    )
    att_mask2 = tf.cast(tf.where(
        tf.nn.sigmoid(decode_attention_out2) > 0.5,
        tf.ones_like(decode_attention_out2),
        tf.zeros_like(decode_attention_out2))*255, tf.uint8
    )

    input_mask1 = tf.image.resize(input_mask, (int(input_shape[0]/2), int(input_shape[1]/2)))
    # mask1_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=decode_attention_out1, labels=input_mask1)
    # )
    mask1_loss = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(logits=decode_attention_out1, targets=input_mask1, pos_weight=100)
    )
    input_mask2 = tf.image.resize(input_mask, (int(input_shape[0]/4), int(input_shape[1]/4)))
    # mask2_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=decode_attention_out2, labels=input_mask2)
    # )
    mask2_loss = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(logits=decode_attention_out2, targets=input_mask2, pos_weight=100)
    )
    logits = tf.squeeze(logits)
    class_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    losses = mask1_loss + mask2_loss + class_loss
    if reg_l2:
        losses += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(logits, 1),
                tf.cast(tf.squeeze(labels), np.int64)
            ),
            np.float
        )
    )

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(
        learning_rate, global_step, 1000, 0.97, staircase=True
    )
    optimizer = tf.train.AdamOptimizer(lr).minimize(losses, global_step=global_step)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=5)

    writer = tf.summary.FileWriter(r'./boarder', sess.graph)
    sum_lr = tf.summary.scalar('a0-learning_rate', lr)
    sum_acc_train = tf.summary.scalar('a1-train_accuracy', acc)
    sum_loss_train = tf.summary.scalar('a2-train_loss', losses)
    sum_loss_attention1_train = tf.summary.scalar('a3-loss_attention_stage1', mask1_loss)
    sum_loss_attention2_train = tf.summary.scalar('a4-loss_attention_stage2', mask2_loss)
    sum_loss_class_train = tf.summary.scalar('a5-loss_class', class_loss)

    sum_ori_img = tf.summary.image('img', input_x, max_outputs=batch_size)
    sum_mask1 = tf.summary.image('mask1', input_mask1, max_outputs=batch_size)
    sum_mask2 = tf.summary.image('mask2', input_mask2, max_outputs=batch_size)
    sum_mask_att1 = tf.summary.image('mask_att1', att_mask1, max_outputs=batch_size)
    sum_mask_att2 = tf.summary.image('mask_att2', att_mask2, max_outputs=batch_size)

    sum_train = tf.summary.merge([
        sum_acc_train, sum_loss_train, sum_loss_attention1_train,
        sum_loss_attention2_train, sum_loss_class_train, sum_lr
    ])
    sum_img = tf.summary.merge([sum_ori_img, sum_mask1, sum_mask2, sum_mask_att1, sum_mask_att2])

    if os.path.exists('./weights/checkpoint'):
        try:
            logger.info('Loading trained model...')
            ckp = tf.train.get_checkpoint_state('./weights')
            saver.restore(sess, ckp.model_checkpoint_path)
            logger.info('Loading trained model success.')
        except Exception:
            logger.error('Error while loading trained model')
            init = tf.global_variables_initializer()
            sess.run(init)
    else:
        logger.info("Can't find trained model.")
        init = tf.global_variables_initializer()
        sess.run(init)

    num_iters = len(reader.labels_list_train) // batch_size
    val_imgs, val_masks, val_labels = reader.get_val_imgs()
    for e in range(epoches):
        pool = Pool(processes=4, maxtasksperchild=None)
        for m_iter in range(num_iters):
            batch_imgs, batch_masks, batch_labels = reader.next_batch(batch_size, pool)
            counter = e * num_iters + m_iter + 1
            _, train_acc_, train_loss_, mask1_loss_, mask2_loss_, class_loss_, sum_train_ = sess.run(
                [optimizer, acc, losses, mask1_loss, mask2_loss, class_loss, sum_train],
                feed_dict={input_x: batch_imgs, input_mask: batch_masks, labels: batch_labels, is_training: True}
            )
            writer.add_summary(sum_train_, counter)
            logger.info(
                '[Train] Epoch[{}/{}] step[{}/{}] train_acc:{} trainLoss:{} mask1Loss:{} mask2Loss:{} classLoss:{}'
                .format(
                    e+1, epoches, m_iter + 1, num_iters, train_acc_, train_loss_, mask1_loss_, mask2_loss_, class_loss_
                )
            )
            # # 每个epoch开始，保存生成的mask
            # if m_iter == 0 and e > 0:
            #     mask1_, mask2_, mask_att1_, mask_att2_ = sess.run(
            #         [input_mask1, input_mask2, decode_attention_out1, decode_attention_out1],
            #         feed_dict={input_x: batch_imgs, input_mask: batch_masks, labels: batch_labels, is_training: False}
            #     )
            #     for i in range(batch_size):
            #         temp_mask1 = mask1_[i]
            #         temp_mask2 = mask2_[i]
            #         temp_mask_att1 = mask_att1_[i]
            #         temp_mask_att2 = mask_att2_[i]
            #         cv2.imwrite(
            #             './datasets/masks/ori1/epoch_{}_mask1_{}.jpg'.format(e, i), (temp_mask1*255).astype(np.uint8)
            #         )
            #         cv2.imwrite(
            #             './datasets/masks/ori2/epoch_{}_mask2_{}.jpg'.format(e, i), (temp_mask2*255).astype(np.uint8)
            #         )
            #         cv2.imwrite(
            #             './datasets/masks/att1/epoch_{}_mask1_{}.jpg'
            #             .format(e, i), (temp_mask_att1*255).astype(np.uint8)
            #         )
            #         cv2.imwrite(
            #             './datasets/masks/att2/epoch_{}_mask2_{}.jpg'
            #             .format(e, i), (temp_mask_att2*255).astype(np.uint8)
            #         )

            # 显示图片，mask，和生成的mask
            if counter % 200 == 0:
                sum_img_ = sess.run(
                    sum_img,
                    feed_dict={input_x: val_imgs[:batch_size], input_mask: val_masks[:batch_size], is_training: False}
                )
                writer.add_summary(sum_img_)
            # 验证集上进行交叉验证
            if counter % 50 == 0:
                num_iter_val = int(np.ceil(len(val_labels)/batch_size))
                val_acc = []
                val_loss = []
                val_mask1_loss = []
                val_mask2_loss = []
                val_class_loss = []
                for it in range(num_iter_val):
                    val_imgs_t = val_imgs[it*batch_size: (it+1)*batch_size]
                    val_masks_t = val_masks[it*batch_size: (it+1)*batch_size]
                    val_labels_t = val_labels[it*batch_size: (it+1)*batch_size]
                    val_acc_, val_loss_, val_mask1_loss_, val_mask2_loss_, val_class_loss_ = sess.run(
                        [acc, losses, mask1_loss, mask2_loss, class_loss],
                        feed_dict={
                            input_x: val_imgs_t, input_mask: val_masks_t, labels: val_labels_t, is_training: False
                        }
                    )
                    val_acc.append(val_acc_)
                    val_loss.append(val_loss_)
                    val_mask1_loss.append(val_mask1_loss_)
                    val_mask2_loss.append(val_mask2_loss_)
                    val_class_loss.append(val_class_loss_)
                val_acc_m = np.mean(val_acc)
                val_loss_m = np.mean(val_loss)
                val_mask1_loss_m = np.mean(val_mask1_loss)
                val_mask2_loss_m = np.mean(val_mask2_loss)
                val_class_loss_m = np.mean(val_class_loss)
                sum_val_ = tf.Summary(value=[
                    tf.Summary.Value(tag="b1-val_acc", simple_value=val_acc_m),
                    tf.Summary.Value(tag="b2-val_loss", simple_value=val_loss_m),
                    tf.Summary.Value(tag="b3-val_mask1_loss", simple_value=val_mask1_loss_m),
                    tf.Summary.Value(tag="b4-val_mask2_loss", simple_value=val_mask2_loss_m),
                    tf.Summary.Value(tag="b5-val_class_loss", simple_value=val_class_loss_m)
                ])
                writer.add_summary(sum_val_, counter)
        pool.close()
        pool.join()
        if (e + 1) % 100 == 0:
            saver.save(sess, r'./weights/model_epoch_{}.ckpt'.format(e))


if __name__ == '__main__':
    import data_reader
    datasets_path = r'./datasets'
    reader = data_reader.DataReader(datasets_path, is_training=True)
    main(reader)
