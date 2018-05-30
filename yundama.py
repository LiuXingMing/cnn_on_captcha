# encoding=utf-8
import uuid
import base64
import requests
import tempfile
import httplib, mimetypes, urlparse, json, time

######################################################################

# 错误代码请查询 http://www.yundama.com/apidoc/YDM_ErrorCode.html
# 所有函数请查询 http://www.yundama.com/apidoc

# 1. http://www.yundama.com/index/reg/developer 注册开发者账号
# 2. http://www.yundama.com/developer/myapp 添加新软件
# 3. 使用添加的软件ID和密钥进行开发，享受丰厚分成

# 用户名
username = '***'		# 填入云打码的账号

# 密码
password = '***'		# 填入云打码的密码

# 软件ＩＤ，开发者分成必要参数。登录开发者后台【我的软件】获得！
appid = 1

# 软件密钥，开发者分成必要参数。登录开发者后台【我的软件】获得！
appkey = '22cc5376925e9387a23cf797cb9ba745'

# 图片文件
filename = 'ab.png'

# 验证码类型，# 例：1004表示4位字母数字，不同类型收费不同。请准确填写，否则影响识别率。在此查询所有类型 http://www.yundama.com/price.html
codetype = 1004

# 超时时间，秒
timeout = 20


######################################################################

class YDMHttp:
    apiurl = 'http://api.yundama.net:5678/api.php'

    username = ''
    password = ''
    appid = ''
    appkey = ''

    def __init__(self, username, password, appid, appkey, filename):
        self.username = username
        self.password = password
        self.appid = str(appid)
        self.appkey = appkey
        self.filename = filename

    def request(self, fields, files=[]):
        try:
            response = post_url(self.apiurl, fields, files)
            response = json.loads(response)
        except Exception as e:
            response = None
        return response

    def balance(self):
        data = {'method': 'balance', 'username': self.username, 'password': self.password, 'appid': self.appid,
                'appkey': self.appkey}
        response = self.request(data)
        if (response):
            if (response['ret'] and response['ret'] < 0):
                return response['ret']
            else:
                return response['balance']
        else:
            return -9001

    def login(self):
        data = {'method': 'login', 'username': self.username, 'password': self.password, 'appid': self.appid,
                'appkey': self.appkey}
        response = self.request(data)
        if (response):
            if (response['ret'] and response['ret'] < 0):
                return response['ret']
            else:
                return response['uid']
        else:
            return -9001

    def upload(self, codetype, timeout):
        data = {'method': 'upload', 'username': self.username, 'password': self.password, 'appid': self.appid,
                'appkey': self.appkey, 'codetype': str(codetype), 'timeout': str(timeout)}
        file = {'file': self.filename}
        response = self.request(data, file)
        if (response):
            if (response['ret'] and response['ret'] < 0):
                return response['ret']
            else:
                return response['cid']
        else:
            return -9001

    def result(self, cid):
        data = {'method': 'result', 'username': self.username, 'password': self.password, 'appid': self.appid,
                'appkey': self.appkey, 'cid': str(cid)}
        response = self.request(data)
        return response and response['text'] or ''

    def decode(self, codetype, timeout):
        cid = self.upload(codetype, timeout)
        if (cid > 0):
            for i in range(0, timeout):
                result = self.result(cid)
                if (result != ''):
                    return cid, result
                else:
                    time.sleep(1)
            return -3003, ''
        else:
            return cid, ''


######################################################################

def post_url(url, fields, files=[]):
    urlparts = urlparse.urlsplit(url)
    return post_multipart(urlparts[1], urlparts[2], fields, files)


def post_multipart(host, selector, fields, files):
    content_type, body = encode_multipart_formdata(fields, files)
    h = httplib.HTTP(host)
    h.putrequest('POST', selector)
    h.putheader('Host', host)
    h.putheader('Content-Type', content_type)
    h.putheader('Content-Length', str(len(body)))
    h.endheaders()
    h.send(body)
    errcode, errmsg, headers = h.getreply()
    return h.file.read()


def encode_multipart_formdata(fields, files=[]):
    BOUNDARY = 'WebKitFormBoundaryJKrptX8yPbuAJLBQ'
    CRLF = '\r\n'
    L = []
    for field in fields:
        key = field
        value = fields[key]
        L.append('--' + BOUNDARY)
        L.append('Content-Disposition: form-data; name="%s"' % key)
        L.append('')
        L.append(value)
    for field in files:
        key = field
        filepath = files[key]
        L.append('--' + BOUNDARY)
        L.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filepath))
        L.append('Content-Type: %s' % get_content_type(filepath))
        L.append('')
        L.append(open(filepath, 'rb').read())
    L.append('--' + BOUNDARY + '--')
    L.append('')
    body = CRLF.join(L)
    content_type = 'multipart/form-data; boundary=%s' % BOUNDARY
    return content_type, body


def get_content_type(filename):
    return mimetypes.guess_type(filename)[0] or 'application/octet-stream'


######################################################################


def identify(filename, codetype=1000):
    if (username == 'username'):
        print '请设置好相关参数再测试'
    else:
        # 初始化
        yundama = YDMHttp(username, password, appid, appkey, filename)

        # 登陆云打码
        uid = yundama.login()
        # print 'uid: %s' % uid

        # 查询余额
        balance = yundama.balance()
        # print 'balance: %s' % balance

        # 开始识别，图片路径，验证码类型ID，超时时间（秒），识别结果
        cid, result = yundama.decode(codetype, timeout)
        # print 'cid: %s, result: %s' % (cid, result)
        return result


def identify2(url, session=requests.Session(), codetype=1000, retry=5, ignoreError=False):
    """ 验证码识别，URL可以是验证码的URL，也可以是验证码的base64 """
    code = ''
    failure = 0
    while failure < retry:
        try:
            if url.startswith('http'):
                r = session.get(url, timeout=15)
                img_txt = r.content
            else:
                img_txt = base64.b64decode(url)
                failure += 2        # 如果是图片的base64，将不能更新图片，最多识别两次。
            picname = '%s/%s.png' % (tempfile.gettempdir(), str(uuid.uuid1()))
            with open(picname, 'wb') as f:
                f.write(img_txt)

            code = identify(picname, codetype=codetype)  # 验证码识别
            if len(code) == 0 or '看不清' in code:
                failure += 1
                continue
            break
        except Exception, e:
            break

    if len(code) == 0 and not ignoreError:
        raise Exception('failed')
    return code
