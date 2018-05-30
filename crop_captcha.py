# encoding=utf-8
# crawl_captcha.py用来采集验证码，此文件用来将验证切分成单个字符

import os
import glob
from PIL import Image


def deal(path):
    filenames = glob.glob(path + '*.png')
    for filename in filenames:
        try:
            uid, code = filename.split('/')[-1].split('.png')[0].split('---')
            img = Image.open(filename)
            if img.size != (60, 46):
                raise Exception('%s error: %s' % (uid, img.size))
            if code is None or len(code) != 4:
                raise Exception('%s error: %s' % (uid, code))

            # 每个字符一个文件夹，如果不存在，则创建
            for i in range(len(code)):
                character = code[i].lower()
                if os.path.exists('./%s/' % character) is False:
                    os.mkdir('./%s/' % character)

                img.crop((6+10*i, 4, 12+10*i, 22)).save('./%s/%s.png' % (character, uid))

            print('Finish %s    %s' % (uid, list(code)))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    deal('./captchas/')



