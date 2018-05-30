# encoding=utf-8

import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from multiprocessing import Pool


number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

IMAGE_HEIGHT = 18
IMAGE_WIDTH = 6
MAX_CAPTCHA = 1
CHAR_SET_LEN = 36


def gen_captcha_text_and_image(filename):
    images = []
    captcha_image = Image.open(filename)
    for i in range(4):
        img = captcha_image.crop((6 + 10 * i, 4, 12 + 10 * i, 22))
        img = np.array(img)
        img = convert2gray(img)
        img = img.flatten() / 255
        images.append(img)
    return images


def vec2text(char_set=number+alphabet, pos=None):
    if isinstance(pos, int):
        return char_set[pos]
    elif isinstance(pos, list):
        word = []
        for one in pos:
            word.append(char_set[one])
        return ''.join(word)


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def crack_captcha_cnn(xx, keep_prob, w_alpha=0.01, b_alpha=0.1):
    """
        定义CNN
        w_alpha, b_alpha传入一个很小的值作为初始化值
    """
    x = tf.reshape(xx, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  # 传入的X为[batch_size,H,W]，需要转换为Tensorflow格式[batch_size,H,W，Chanel]

    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))  # filter:3*3,输入通道1(灰度图)，输出（特征图）：32
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.dropout(conv3, keep_prob)

    w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 128]))
    b_c4 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
    conv4 = tf.nn.dropout(conv4, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([6 * 18 * 128, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    # 卷积结果扁平化
    dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def crack_captcha(captcha_image):
    xx = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    yy = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float32)  # dropout
    output = crack_captcha_cnn(xx, keep_prob)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/crack_capcha.model-60000")

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={xx: [captcha_image], keep_prob: 1})
        vec = text_list[0].tolist()
        sess.close()
        return vec2text(pos=vec)


def identify(filename):
    code = filename.split('/')[-1].split('.png')[0].split('---')[-1]
    images = gen_captcha_text_and_image(filename)
    pool = Pool(4)
    result = pool.map(crack_captcha, images)
    pool.close()
    pool.join()
    predict_text = ''.join(result)
    isTrue = predict_text.lower() == code.lower()
    print('%s: %s --> %s' % ('successful' if isTrue else 'failed', code, predict_text))
    return isTrue


if __name__ == '__main__':
    files = glob.glob('./test/*.png')
    result = []
    for file in files:
        result.append(identify(file))
    print('finished: %.2f%%' % (result.count(True) * 100.0 / len(result)))



