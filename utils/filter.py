from dataclasses import replace
import difflib
import re
from zhon.hanzi import punctuation
import string
from ltp import LTP
import json
import requests

# ---------------------------------用于筛选的一些函数-----------------------------------------

def exist_num(text):
    return re.search(r'[1-9]',text)

def is_Chinese(w):
    if '\u4e00' <= w <= '\u9fff':
        return True
    else:
        return False

def is_zh_punctuation(w):
    punctuation_str = punctuation   #中文符号
    if w in punctuation_str:
        return True
    else:
        return False

def is_en(w):
    if 'a'<=w<='z' or 'A'<=w<='Z':
        return True
    else:
        return False

def is_en_punctuation(w):
    punctuation_string = string.punctuation
    if w in string.punctuation:
        return True
    else:
        return False


def exist_marks(text):
    marks = "，。？、！；：/"
    for m in marks:
        if m in text:
            return True
    return False

def delete_detected_error(alerts):
    res = []
    for alert in alerts:
        if alert['suggestion'] == '删除':
            if len(alert['source']) <= 3:
                if not exist_marks(alert['source']):
                    res.append(alert)
    return res


def exist_no_ch(text):
    if len(text) != 0:
        if re.search('[\u4e00-\u9fa5]',text):
            return False
        else:
            return True
    else:
        return False

def exist_ri(source,fixed):
    if re.search('日',source) and re.search('曰',fixed):
        return True
    return False

def content_in_marks(alert,oritext):
    ori_content = oritext
    start=alert["start"]
    content_befor = ori_content[:start]
    mark_dict={
        "“":"”",
        "《":"》",
        "（":"）",
        "【":"】"
    }
    left,right=0,0
    for key,value in zip(mark_dict.keys(),mark_dict.values()):
        left=content_befor.count(key)
        right=content_befor.count(value)
        if left>right:#不处理
            return True
            # item.remove(alert)
        
    return False


def exist_change_zhnum(alert,oritext):
    start=alert["start"]
    end=alert["end"]
    content = oritext[start:]
    ptnNumberCN = re.compile('([Ｏ零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟]+)([年月日桶元圆角分])')
    if ptnNumberCN.search(content):#不处理
        return True
    return False

def exist_leader(alert,oritext,ltp):
    spec_name_txt = open('/home/qian/zhouxu/work_space/pycorrector/pycorrector/t5/dict/spec_name.txt',encoding='utf-8')
    spec_name_list = [line.replace('\n','') for line in spec_name_txt]
    ltp.add_words(spec_name_list)
    sourceText = alert['sourceText']
    replaceText = alert['replaceText']
    start = alert['start']
    if replaceText in spec_name_list:
        return True
    wrong_sen = oritext[start:]
    wrong_sen_list = ltp.pipeline(wrong_sen)
    if wrong_sen_list[0][0] in spec_name_list:
        return True
    return False

def exist_nones(alert,ltp):
    n_list = ltp.pipeline([alert['sourceText']], tasks=["cws", "pos"]).pos
    if 'nh' in n_list[0] or 'ns' in n_list[0] or 'nz' in n_list[0] or 'j' in n_list[0]:
        return True
    return False

def second_filter(alerts,oritext,ltp,filt):
    ''' 
    用于在送入后处理前进行一次过滤:
    过滤功能如下：
    1.存在习**的问题
    2.存在日曰的问题
    3.存在中文数字的问题
    4.存在人名的问题
    5.存在修改内容与原文相差过大的问题
    '''
    
    res = []
    for alert in alerts:
        if exist_no_ch(alert['sourceText']) or exist_no_ch(alert['replaceText']):     #不存在中文
            continue
        if exist_ri(alert['sourceText'],alert['replaceText']):#存在日曰
            continue
        if content_in_marks(alert,oritext):
            continue
        # if exist_leader(alert,oritext,ltp):#存在领导
        #     continue
        if exist_change_zhnum(alert,oritext):#存在中文数字
            continue
        if exist_nones(alert,ltp):#存在名词
            continue
        if alert['sourceText'] == alert['replaceText']:
            continue
        if filt.filter(alert):
            continue
        if abs(len(alert['sourceText'])-len(alert['replaceText'])) < 4:
            res.append(alert)
    return res

def allChar_ch(text):
    for c in text:
        if not is_Chinese(c):
            return False
    return True

def first_filter(type,sourceText,replaceText):
    # 明确一个事情，我们的功能只能修改中文，用于生成json前的提一次过滤
    if type == 'replace':
        if allChar_ch(sourceText) and allChar_ch(replaceText):
            return False
        else:
            return True
    elif type == 'insert':
        if allChar_ch(replaceText):
            return False
        else:
            return True
    elif type == 'delete':
        if allChar_ch(replaceText):
            return False
        else:
            return True
    else:
        True


def t5(data):
    header = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }
    url = 'http://47.94.233.147:8089/corrector'
    data = json.dumps(data)
    r = requests.post(url=url, data=data, headers=header)
    result = json.loads(r.text)
    return result

if __name__ == '__main__':
    data = {
        "texts":
            [
                "在方舱里，他每个班次平均要运送四五十桶垃圾。"
            ]
    }
    ltp =LTP('LTP/base1')
    result=t5(data)
    alerts = result['alerts'][0]
    sentence = result['data']
    oritext = sentence[0]['originText']
    print(second_filter(alerts,oritext,ltp))