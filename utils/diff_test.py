#coding:utf-8
import difflib
# from ltp import LTP
import pandas
import re
# import jieba

from zhon.hanzi import punctuation
import string
import sqlite3
from loguru import logger
from utils.checkAdvise import SimComputer
from utils.filter import allChar_ch
from utils.fixModeloutput import long2short
# from paddlenlp import Taskflow

ltp = LTP('LTP/base1')

def create_json(item):
    if item['originText'] != item['output']:
        model_json = compare_text(item['originText'],item['output'])
        model_json = supple_error(model_json,item['originText'])
        return model_json
    else:
        return []

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

def exist_specialmark(text):
    for c in text:
        if not is_Chinese(c) and not is_en(c):
            return True
        else:
            return False

def exist_ChangeUnit(source):
    return  re.search('[零Ｏ二三四五六七八九十千百万亿]+[元年月日]',source)

# 测试用
def run_db2dict_LTP(dbdir):
    # 将db中的一些词添加到分词库里
    specialWord = []
    con = sqlite3.connect(dbdir)
    cur = con.cursor()
    cur.execute('select source_text from black_pairs')
    for row in cur:
        specialWord.append(row[0])
    cur.execute('select word from black_words')
    for row in cur:
        specialWord.append(row[0])
    l = len(specialWord)

    ltp.add_words(specialWord)
    logger.info(f'specialWord has append into LTP dict, len(specialWord)=={l}')

# def run_db2dict(dbdir):
#     # 将db中的一些词添加到分词库里
#     specialWord = []
#     con = sqlite3.connect(dbdir)
#     cur = con.cursor()
#     cur.execute('select source_text from black_pairs')
#     for row in cur:
#         specialWord.append(row[0])
#     cur.execute('select word from black_words')
#     for row in cur:
#         specialWord.append(row[0])
#     l = len(specialWord)

#     jieba.initialize()
#     for word in specialWord:
#         jieba.add_word(word)
#     logger.info(f'specialWord has append into jieba dict, len(specialWord)=={l}')


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


def supple_error_LTP(errors,sentence):
    # 用于扩展错误的信息
    # 首先是切分句子
    sentence = sentence.encode('utf-8', errors='ignore').decode('utf-8')
    cut_text,cut_mark = long2short(sentence)
    result = ltp.pipeline(cut_text,tasks=['cws'])
    texts = result.cws
    textlist = []
    for m in cut_mark:
        if isinstance(m,int):
            textlist.extend(texts[m])  
        else:
            textlist.append(m)
    # 测试用
    # s = ''
    # for t in textlist:
    #     s += t
    # diff = compare_text(sentence,s)
    # 然后是根据切分的句字和error信息来扩展error
    textlenth = [len(text) for text in textlist]
    res = []
    
    for error,type in errors:
        start = error['start']
        end = error['end'] + 1
        ori = error['sourceText']
        fix = error['replaceText']
        
        num = 0
        i = 0
        while num + textlenth[i] <= start and i+1<len(textlenth):
                num += textlenth[i]
                i += 1
        
        if i+1 == len(textlist):
            base = len(sentence)-textlenth[i]
            error['start'] = base
            error['sourceText'] = textlist[i]
            source = textlist[i]
            error['end'] = len(sentence)
        else:
            base = num
            error['start'] = num
            
            j = i
            while num + textlenth[j] < end and j+1<len(textlenth):
                num += textlenth[j]
                j += 1
            

            source = ''
            
            for t in range(i,j+1):
                source += textlist[t]
            error['sourceText'] = source
            error['end'] = num + textlenth[j]
            
            if j+1 == len(textlist):
                error['end'] = len(sentence)

        if type == 'replace':
            x = source[:start-base] + fix + source[end-base:]
        elif type == 'delete':
            if j+1 <len(textlenth):
                source = ''
                for t in range(i,j+2):
                    source += textlist[t]
                error['sourceText'] = source
                error['end'] = num + textlenth[j] + textlenth[j+1]
            else:
                source = sentence[base:]
                error['sourceText'] = source
                error['end'] = len(sentence)

            x = source[:start-base] + source[end-base:]
        else:
            if error['alertType'] == 1:
                # 向前增加
                x = source[:start-base] + fix +source[start-base:]
            else:
                # 向后增加
                x = source[:end-base] + fix +source[end-base:]


        error['replaceText'] = x
        
        res.append(error)
    return res

# def supple_error(errors,sentence):
#     textlist = jieba.lcut(sentence)
#     textlenth = [len(text) for text in textlist]
#     res = []
    
#     for error,type in errors:
#         start = error['start']
#         end = error['end'] + 1
#         ori = error['sourceText']
#         fix = error['replaceText']
        
#         num = 0
#         i = 0
#         while num + textlenth[i] < start:
#                 num += textlenth[i]
#                 i += 1
        
#         base = num
#         error['start'] = num
        
#         j = i
#         while num + textlenth[j] < end and j+1<len(textlist):
#             num += textlenth[j]
#             j += 1
        

#         source = ''
        
#         for t in range(i,j+1):
#             source += textlist[t]
#         error['sourceText'] = source
#         error['end'] = num + textlenth[j]
        
#         if j+1 == len(textlist):
#             error['end'] = len(sentence)

#         if type == 'replace':
#             x = source[:start-base] + fix + source[end-base:]
#         elif type == 'delete':
#             if j+1 <len(textlenth):
#                 source = ''
#                 for t in range(i,j+2):
#                     source += textlist[t]
#                 error['sourceText'] = source
#                 error['end'] = num + textlenth[j] + textlenth[j+1]
#             else:
#                 source = sentence[base:]
#                 error['sourceText'] = source
#                 error['end'] = len(sentence)

#             x = source[:start-base] + source[end-base:]
#         else:
#             if error['alertType'] == 1:
#                 # 向前增加
#                 x = source[:start-base] + fix +source[start-base:]
#             else:
#                 # 向后增加
#                 x = source[:end-base] + fix +source[end-base:]


#         error['replaceText'] = x
        
#         res.append(error)
#     return res

def compare_text(origin_text, output_text):
    s = difflib.SequenceMatcher(None, origin_text, output_text)
    model_json= []
    
    for (type, ori_pos_begin, ori_pos_end, out_pos_begin, out_pos_end) in s.get_opcodes():
        if type == 'equal':
            continue
        
        sourceText = origin_text[ori_pos_begin: ori_pos_end]
        replaceText = output_text[out_pos_begin: out_pos_end]
        alertMessage=""
        alertType,errorType=-1,-1
        ori_pos_end=ori_pos_end-1
        
        if first_filter(type,sourceText,replaceText):
            continue
        else:
            
            if "*" in replaceText:
                alertMessage="含有敏感词"
                errorType= 6
                alertType = 4
                alert_item = creat_alert_item(alertMessage, alertType, errorType, replaceText, sourceText, ori_pos_begin,
                                            ori_pos_end)
                model_json.append([alert_item,type])
                continue
            #替换
            elif type == "replace":
                alertMessage = f"建议用“{replaceText}”替换“{sourceText}”"
                alertType = 4
                errorType = 1
                alert_item = creat_alert_item(alertMessage, alertType, errorType, replaceText, sourceText, ori_pos_begin,
                                            ori_pos_end)
                model_json.append([alert_item,type])
                continue
            elif type == "insert":
                alertMessage = f"建议添加“{replaceText}”"
                #判断添加位置，抽取添加字符的位置，和原文进行比较
                if ori_pos_end<ori_pos_begin:
                    ori_pos_end=ori_pos_begin
                alertType = judge_position(origin_text,ori_pos_begin,output_text,out_pos_end)
                errorType = 5
                alert_item = creat_alert_item(alertMessage, alertType, errorType, replaceText, sourceText, ori_pos_begin,
                                            ori_pos_end)
                model_json.append([alert_item,type])
                continue
            elif type == "delete":
                alertMessage = f"建议删除“{sourceText}”"
                alertType = 3
                errorType = 5
                alert_item = creat_alert_item(alertMessage, alertType, errorType, replaceText, sourceText, ori_pos_begin,
                                            ori_pos_end)
                model_json.append([alert_item,type])
                continue
    return model_json

def judge_date(original_text,output_text,ori_pos_begin, ori_pos_end):
    ori_out_list = []
    ori_message_list = extract_message(original_text, type='时间类_具体时间')
    out_message_list = extract_message(output_text, type='时间类_具体时间')
    # print(ori_message_list)
    # print(out_message_list)
    for ori_message,out_message in zip(ori_message_list,out_message_list):
        
        if ori_pos_begin>=ori_message[1] and ori_pos_end<=ori_message[2]:
            ori_out_list.append(ori_message)
            ori_out_list.append(out_message)
            
    return  ori_out_list
    
def judge_area(original_text,output_text,ori_pos_begin, ori_pos_end):
    ori_out_list = []
    ori_message_list = extract_message(original_text, type='世界地区类')
    out_message_list = extract_message(output_text, type='世界地区类')

    # print(ori_message_list)
    # print(out_message_list)
    for ori_message, out_message in zip(ori_message_list, out_message_list):
        
        if ori_pos_begin >= ori_message[1] and ori_pos_end <= ori_message[2]:
            ori_out_list.append(ori_message)
            ori_out_list.append(out_message)
           
    return ori_out_list
#先判断有无人名，有人名抽取职位，如果职位抽取不为0，则判断人名和职位是否连在一起，如果连在一起，则判断该人名职位是否正确
def judge_name(original_text,output_text,ori_pos_begin, ori_pos_end):
    ori_out_list = []
    ori_message_list = extract_message(original_text, type='人物类_实体')
    out_message_list = extract_message(output_text, type='人物类_实体')
    for ori_message, out_message in zip(ori_message_list, out_message_list):
        if ori_pos_begin >= ori_message[1] and ori_pos_end <= ori_message[2]:#是人名错误，判断修改后的人名是否在表中
            #人名错误，首先判断是否有职务，然后判断职位是否正确。
            leader_name=out_message[0]
            if judge_is_leader(leader_name):#是领导，判断职务
                ori_out_list.append(ori_message)
                ori_out_list.append(out_message)
    return ori_out_list

def judge_is_leader(leader_name):
    path = "0630 领导人更新.xls"
    leader_info = pandas.read_excel(path)
    for name in leader_info["领导人名称"]:
        if type(name) == float:
            name = str(name)
        name = name.replace("\n", "")

    name_set=set(leader_info["领导人名称"])
    if leader_name in name_set:
        return True
    else:
        return False
           
def judge_position(origin_text,origin_pos_begin,output_text,out_pos_end):
    alertType=-1
    origin_pos_end1=origin_pos_begin+3
    if origin_pos_end1>=len(origin_text):
        origin_pos_end1=len(origin_text)-1
    origin_content=origin_text[origin_pos_begin:origin_pos_end1]
    print(origin_content)
    print(output_text[out_pos_end:])
    if origin_content in output_text[out_pos_end:]:
        alertType=2
    else:
        alertType=1
    return alertType

def judge_mark(error,correct_error):
    sentence_mark = {
        ",": "，",
        ".": "。",
        ":": "：",
        "?": "？",
        "!": "！",
        "\'": "‘",
        "\"": "“",
        "(": "（",
        "<": "《"
    }
    for key,value in zip(sentence_mark.keys(),sentence_mark.values()):
        if error==key and correct_error==value:#英文标点改成了中文标点
            return True
        elif error==value and correct_error==key:#中文标点改成了英文标点:
            return False

def extract_message(sentence: str,type="世界地区类"):
    length=0
    namelist=[]
    lac = Taskflow("ner")
    result = lac(sentence)
    print(result)
    for pair in result:
        name_list=[]
        length+=len(pair[0])
        if pair[1] in type:
            #判断为人物名字
            start=length-len(pair[0])
            end=length-1
            name_list.append(pair[0])
            name_list.append(start)
            name_list.append(end)
            namelist.append(name_list)
    return namelist

def creat_alert_item(alertMessage, alertType, errorType, replaceText, sourceText, ori_pos_begin, ori_pos_end):

    if errorType > 100:
        advancedTip = "true"
    elif errorType <= 100:
        advancedTip = "false"
    if errorType != -1 and alertType != -1:
        alert_item = item(advancedTip, alertMessage, alertType, errorType, replaceText, sourceText,
                                      ori_pos_begin, ori_pos_end)
    return  alert_item
def item(advancedTip, alertMessage, alertType, errorType, replaceText, sourceText, ori_pos_begin, ori_pos_end):
    res = {
        'advancedTip': advancedTip,
        'alertMessage': alertMessage,
        'alertType': alertType,
        'errorType': errorType,
        'replaceText': replaceText,
        'sourceText': sourceText,
        'start': ori_pos_begin,
        'end': ori_pos_end,
    }
    return res

if __name__ == '__main__':
    model_json = {}
    origin_text = "二Ｏ二一年"
    output_text ="二O二一年"
    type = 'replace'
    first_filter(type,origin_text,output_text)
    # header = {
    #     'Content-Type': 'application/json',
    #     'accept': 'application/json'
    # }
    # data = {"texts": ["2022年8月32日。"]}
    # url = 'http://47.94.233.147:8089/corrector'
    # data = json.dumps(data)
    # r = requests.post(url=url, data=data, headers=header)
    # result = json.loads(r.text)
    # print(result)
    
    # model_json = compare_text(origin_text, output_text, model_json)
    # print(model_json)
    # print(judge_date(origin_text,output_text,"2022年8月3","2022年8月3"))
    # location_str = [
    #     "广东省深圳市福田区巴丁街深南中路1025号新城大厦1层",
    #     "特斯拉上海超级工厂是特斯拉汽车首座美国本土以外的超级工厂，位于中华人民共和国上海市。",
    #     "三星堆遗址位于中国四川省广汉市城西三星堆镇的鸭子河畔，属青铜时代文化遗址"
    # ]
    
    # print(judge_area(origin_text,output_text,8,8))
    # model_json = compare_text(origin_text,output_text)
    # model_json = supple_error_LTP(model_json,origin_text)
    # model_json = json.dumps(model_json,indent=4,ensure_ascii=False)
    # print(model_json)

