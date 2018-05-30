# encoding=utf-8
# 功能：采集学库宝的验证码，使用时请先注册云打码的账号，将账号密码填入 yundama.py 。

import os
import time
import shutil
import requests
import datetime
from uuid import uuid1
from yundama import identify
from multiprocessing import Pool


def crawl(_):
    filename = '%s.png' % str(uuid1())
    try:
        session = requests.Session()
        session.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Host': 'm.xuekubao.com',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Linux; U; Android 2.3.6; en-us; Nexus S Build/GRK39F) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1',
        }
        session.get('http://m.xuekubao.com/shiti/710240.shtml')

        session.headers = {
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Host': 'm.xuekubao.com',
            'Referer': 'http://m.xuekubao.com/shiti/710240.shtml',
            'User-Agent': 'Mozilla/5.0 (Linux; U; Android 2.3.6; en-us; Nexus S Build/GRK39F) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1'
        }
        r = session.get('http://m.xuekubao.com/index.php?m=Show&a=verify&tid=710240', timeout=10)
        if r.status_code != 200:
            print 'status_code: %s' % r.status_code
            time.sleep(2)
            return
        with open(filename, 'wb') as f:
            f.write(r.content)

        code = identify(filename, codetype=1004)
        if len(code) != 4:
            raise Exception('identify failed.')

        data2 = {
            'code': code,
            'tid': '710240'
        }
        session.headers = {
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Host': 'm.xuekubao.com',
            'Origin': 'http://m.xuekubao.com',
            'Referer': 'http://m.xuekubao.com/shiti/710240.shtml',
            'User-Agent': 'Mozilla/5.0 (Linux; U; Android 2.3.6; en-us; Nexus S Build/GRK39F) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1',
            'X-Requested-With': 'XMLHttpRequest'
        }
        r2 = session.post('http://m.xuekubao.com/index.php?m=Show&a=imgcheck', data=data2, timeout=10)
        if '{"error":1}' not in r2.content:
            shutil.move(filename, './captcha/%s---%s.png' % (filename.split('.')[0], code))
            print '%s: successful: %s --> %s' % (datetime.datetime.now(), filename, code)
        else:
            print 'failed: %s' % filename
    except Exception, e:
        print e
    finally:
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == '__main__':
    pool = Pool(1)
    pool.map(crawl, range(3000))
    pool.close()
    pool.join()

