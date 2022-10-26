# -*- coding:utf-8 -*-
import json
import re
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from loguru import logger
import time
import json
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cut_s(s):
    s = re.sub('([""''])', '', s)
    s = re.sub('[  \001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\u0000\u3000\xa0\xa0]+', '', s)
    return s 

def read_LogFile(logpath):
    '''
    功能：读取log文件，按照以下结构返回一个log信息的list
    logdata = [
        {
            'source': sentence,
            'date':date
            'alerts':alerts
            'tag':True
            'fixed':fixed
        },
        ...
    ]
    sentence ，为输入的语句
    date,为当前log的日期
    alert，为模型返回信息
    tag, 为该句是否需要修改，有修改为True，无修改为False
    fixed,为修正后的句子
    '''
    logfile = open(logpath,encoding='utf-8')
    tag_list = {}
    for log_line in logfile:
        id = re.search(r'[0-9]{8,15}\.[0-9]+',log_line)
        if id==None:
            continue
        if id.group() not in tag_list.keys():
            tag_list[id.group()] = {}
        if '===>' in log_line:
            sen_id = re.search(r'[0-9]{8,15}\.[0-9]+',log_line)
            sen_date = re.search(r'2022-[0-9]{2}-[0-9]{2}',log_line)
            sen_source = log_line.split('===>')
            sen_source = json.loads(sen_source[1])
            try:
                tag_list[sen_id.group()]['sentence'] = sen_source['sentences'][0]
                tag_list[sen_id.group()]['date'] = sen_date.group()
            except KeyError:
                continue
        elif '====' in log_line:
            alert_id = re.search(r'[0-9]{8,15}\.[0-9]+',log_line)
            sen_alerts = log_line.split('====')
            sen_alerts = json.loads(sen_alerts[1])
            try:
                tag_list[alert_id.group()]['alerts'] = sen_alerts['alerts']
            except KeyError:
                continue
            if len(tag_list[alert_id.group()]['alerts'][0]) == 0:
                tag_list[alert_id.group()]['tag'] = False
            else:
                tag_list[alert_id.group()]['tag'] = True
        else:
            continue
    tag_dict = [i for i in tag_list.values() if len(i)>2]
    #print(tag_dict)
    return tag_dict

def catch_RevisedSentence(tag_dict1):
    '''
    功能：输入source,alerts，返回纠正后的句子fixed
    '''
    model_json=[]
    for tag_dict in tag_dict1:
        # print(tag_dicts)
        tag_dict['sentence']= cut_s(tag_dict['sentence'])
        source=tag_dict['sentence']
        alerts=tag_dict['alerts']
        tag=tag_dict['tag']
    # for source,alerts,tag in zip(tag_dict['sentence'],tag_dict['alerts'],str(tag_dict['tag'])):
        if not tag:#不修改
            tag_dict['fixed']=source
            tag_dict['alerts'] = copy.deepcopy(alerts[0])
        else:#需要进行修改
            ori_alert=copy.deepcopy(alerts[0])
            tag_dict['alerts'] = ""
            for item in alerts[0]:#{}
                start=item['start']
                end=item['end']
                if "replaceText" in item.keys():
                    replaceText=item['replaceText']
                else:
                    replaceText=""
                if item['errorType']==5 and item['alertType']==4:  #api认定句子可能存在语法错误，并进行整句替换
                    continue
                if item['alertType']==4 or item['alertType']==3:#替换
                    ori_content=source[start:end+1]
                    source=source[:start]+replaceText+source[end+1:]
                    change_len=len(replaceText)-len(ori_content)
                elif item['alertType']==1:#向前新增
                    source = source[:start] + replaceText + source[start:]
                    change_len = len(replaceText)
                elif item['alertType']==2:#向后新增
                    source = source[:end + 1] + replaceText + source[end+1:]
                    change_len = len(replaceText)
                elif item['alertType']==16:
                    continue
                if change_len!=0:
                    change(change_len, alerts[0], end)
            tag_dict['alerts']=ori_alert
            tag_dict['fixed']=source
        model_json.append(tag_dict)
    return model_json


def change(change_len, alert, index):
    for item in alert:
        start = item["start"]
        end = item["end"]
        if start >= index:
            item['start'] += change_len
            item['end'] += change_len

def saveJson(data,filePath):
    json_dicts = json.dumps(data,indent=4,ensure_ascii=False)
    with open(filePath,'w+',encoding='utf8') as fp:
        fp.write(json_dicts)

class T5Corrector(object):
    def __init__(self, model_dir='t5_best'):
        self.name = 't5_corrector'
        model_dir = model_dir
        bin_path = os.path.join(model_dir, 'pytorch_model.bin')
        logger.warning(f'local model {bin_path} exists, use  model {model_dir}')
        t1 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.model.to(device)
        logger.debug("Use device: {}".format(device))
        logger.debug('Loaded t5 correction model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))

    def batch_t5_correct(self, texts, max_length: int = 128):
        """
        句子纠错
        :param texts: list[str], sentence list
        :param max_length: int, max length of each sentence
        :return: list, (corrected_text, [error_word, correct_word, begin_pos, end_pos])
        """
        result = []
        inputs = self.tokenizer(texts, padding=True, max_length=max_length, truncation=True,
                                return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        for i, text in enumerate(texts):
            text_new = ''
            decode_tokens = self.tokenizer.decode(outputs[i]).replace('<pad>', '').replace('</s>', '').replace(' ',
                                                                                                               '').replace(
                '<unk>', '')
            corrected_text = decode_tokens
            corrected_text = get_errors(corrected_text, text)
            text_new += corrected_text
            # sub_details = [(i[0], i[1], idx + i[2], idx + i[3]) for i in sub_details]
            # details.extend(sub_details)
            result.append(text_new)
        return result


def get_t5Pred(correct_fn, sources):
    res = correct_fn(sources)
    return res

def E_trans_to_C(string):
    E_pun = u',!?()<>:;'
    C_pun = u'，！？（）《》：；'
    table= {ord(f):ord(t) for f,t in zip(E_pun,C_pun)}
    return string.translate(table)

def get_errors(corrected_text, origin_text):
    corrected_text = E_trans_to_C(corrected_text)
    return corrected_text


def result_t5(data,batch_num=200):
    model = T5Corrector('/ddnstor/imu_zhanghuaiwen/mmc_corrector/T5/output/t5_221009_newdata')
    if len(data):
        if len(data) < batch_num:
            res = get_t5Pred(model.batch_t5_correct, data)
        else:
            res = []
            num = math.ceil(len(data)/batch_num)
            for i in range(num):
                if i+1 == num:
                    temp = data[i*batch_num:]
                else:
                    temp = data[i*batch_num:(i+1)*batch_num]
                tempp = get_t5Pred(model.batch_t5_correct,temp)
                res.extend(tempp)
    else:
        res = data
    return res

def diff_ignore(source,fixed,t5_pred,situation_num):
    if situation_num == 2:
        # api认为正确，T5认为错误-->排除T5修改标点符号的影响（主要是T5不处理符号，都是后来增加的）
        pass
    if situation_num == 3:
        # api认为错误，T5认为正确-->排除api修改标点符号的影响
        pass


def eval_log(path, result_path,):
    '''
    使用现有模型对log的source进行预测，比对T5预测结果和api预测结果
    注意：对比的时候最好去除掉空格fixed中的空格再比对，因为T5会删掉空格，可以使用cut_s方法去删空格

    一共会有如下几种情况
    1.api认为正确，T5认为正确->source归为一类保存为logRightSentence_{log日期}.txt
    2.api认为正确，T5认为错误->source和fixed构成训练用的数据集保存为logTrain_{log日期}.json(数据集格式参考以前的训练格式)
    3.api认为错误，T5认为正确->source和fixed构成训练用的数据集保存为logTrain_{log日期}.json(数据集格式参考以前的训练格式)
    4.api认为错误，T5认为错误,且修改一致->fixed归为一类保存为logRightSentence_{log日期}.txt
    5.api认为错误，T5认为错误,修改并不一致->source和fixed构成训练用的数据集保存为logTrain_{log日期}.json(数据集格式参考以前的训练格式)

    将5种情况的对应数量统计一下,计算每一种情况的占比，把第五种情况的例子也统计下来（包括source，fixed，T5_pred）,把这三项保存为一个log_eval_{log日期}.tsv

    '''
    version = path.split('/')[-1].split('.')[0].split('_v')[-1]
    tag_dict1 = read_LogFile(path)
    model_json = catch_RevisedSentence(tag_dict1)
    print(len(model_json))
    logTrain_json = {}
    t5_input = [i['sentence'] for i in model_json]
    t5_output = result_t5(t5_input)
    situation=[[],[],[],[],[]]

    for item,t5_pred in zip(model_json,t5_output):
        source = item['sentence']
        fixed = item['fixed']

        if source == fixed:
            if source == t5_pred:
                #api认为正确，T5认为正确
                flag = 'api认为正确，T5认为正确'
                temp = {
                    "original_text": source,
                    "correct_text": fixed,
                    "flag":flag
                }
                situation[0].append(temp)
            else:
                #api认为正确，T5认为错误
                flag = "api认为正确，T5认为错误"
                temp = {
                    "original_text": source,
                    "correct_text": fixed,
                    "t5_pred":t5_pred,
                    "flag":flag
                }
                situation[1].append(temp)
        else:
            if source == t5_pred:
                # api认为错误，T5认为正确
                flag = "api认为错误，T5认为正确"
                temp = {
                    "original_text": source,
                    "correct_text": fixed,
                    "t5_pred":t5_pred,
                    "flag":flag
                }
                situation[2].append(temp)
            else:
                if fixed == t5_pred:
                    # api认为错误，T5认为错误,且修改一致
                    flag = "api认为错误，T5认为错误,且修改一致"
                    temp = {
                        "original_text": source,
                        "correct_text": fixed,
                        "t5_pred":t5_pred,
                        "flag":flag
                    }
                    situation[3].append(temp)
                else:
                    # api认为错误，T5认为错误,修改并不一致
                    flag = "api认为错误，T5认为错误,修改并不一致"
                    temp = {
                        "original_text": source,
                        "correct_text": fixed,
                        "t5_pred":t5_pred,
                        "flag":flag
                    }
                    situation[4].append(temp)

    # 评价模型现状
    all = len(model_json)
    right = len(situation[0])+len(situation[2])
    eval_result = '''
    1.api认为正确，T5认为正确->{}
    2.api认为正确，T5认为错误->{}
    3.api认为错误，T5认为正确->{}
    4.api认为错误，T5认为错误,且修改一致->{}
    5.api认为错误，T5认为错误,修改并不一致->{}
    总log样例数{}，其中测试模型与api预测结果一致为{}，占比{}
    '''.format(len(situation[0]),len(situation[1]),len(situation[2]),len(situation[3]),len(situation[4]),all,right,(right/all))

    # 统计正确文本
    rightsentence = []
    for item in situation[0]:
        rightsentence.append(item['original_text'])
    for item in situation[2]:
        rightsentence.append(item['correct_text'])
    
    with open(os.path.join(result_path,f'logRightSentence_{version}.txt'),'w+',encoding='utf8') as f:
        for sentence in rightsentence:
            f.write(sentence)
            f.write('\n')
    
    # 其他情况统计
    saveJson(situation[1],os.path.join(result_path,'wrong_situation2.json'))
    saveJson(situation[2],os.path.join(result_path,'wrong_situation3.json'))
    saveJson(situation[4],os.path.join(result_path,'wrong_situation5.json'))

    with open(os.path.join(result_path,f'log_eval_{version}.txt'),'w+',encoding='utf8') as f:
        f.write(eval_result)
    
    return eval_result


def process(path, result_path,result_num):
    '''
    使用现有模型对log的source进行预测，比对T5预测结果和api预测结果
    注意：对比的时候最好去除掉空格fixed中的空格再比对，因为T5会删掉空格，可以使用cut_s方法去删空格

    一共会有如下几种情况
    1.api认为正确，T5认为正确->source归为一类保存为logRightSentence_{log日期}.txt
    2.api认为正确，T5认为错误->source和fixed构成训练用的数据集保存为logTrain_{log日期}.json(数据集格式参考以前的训练格式)
    3.api认为错误，T5认为正确->source和fixed构成训练用的数据集保存为logTrain_{log日期}.json(数据集格式参考以前的训练格式)
    4.api认为错误，T5认为错误,且修改一致->fixed归为一类保存为logRightSentence_{log日期}.txt
    5.api认为错误，T5认为错误,修改并不一致->source和fixed构成训练用的数据集保存为logTrain_{log日期}.json(数据集格式参考以前的训练格式)

    将5种情况的对应数量统计一下,计算每一种情况的占比，把第五种情况的例子也统计下来（包括source，fixed，T5_pred）,把这三项保存为一个log_eval_{log日期}.tsv

    '''
    tag_dict1 = read_LogFile(path)
    model_json = catch_RevisedSentence(tag_dict1)
    print(model_json)
    logTrain_json = {}
    t5_input = [i['sentence'] for i in model_json]
    t5_output = result_t5(t5_input)
    t5y_apiy,t5n_apin_s,t5n_apin_d,t5y_apin,t5n_apiy=0,0,0,0,0
    for item in range(len(model_json)):
        print(t5_input[item])
        logTrain_dict = {}
        if model_json[item]['date'] not in logTrain_json.keys():
            logTrain_json[model_json[item]['date']] = []
        else:
            pass
        logRightSentence = open(result_path + f'/logRightSentence_{model_json[item]["date"]}.txt', 'a', encoding='utf-8')
        # logTrain = open(f'D:\PycharmProjects\pytorch\pycorrector\王一飞\process/logTrain_{item["date"]}.json','a',encoding='utf-8')
        if t5_input[item] == t5_output[item] and model_json[item]['tag']==False: #t5认为正确，api认为正确
            logRightSentence.write(model_json[item]['sentence'] + '\n')
            t5y_apiy += 1
        elif t5_input[item] != t5_output[item] and model_json[item]['tag']==True: #t5认为错误，api认为错误
            # logRightSentence.write(item['sentence'] + '\n')
            if t5_output[item] == model_json[item]['fixed']: #t5与api修改相同
                logRightSentence.write(model_json[item]['sentence'] + '\n')
                t5n_apin_s += 1
            else:  #t5与api修改不相同
                logTrain_dict["original_text"] = model_json[item]['sentence']
                logTrain_dict["correct_text_api"] = model_json[item]['fixed']
                logTrain_dict["correct_text_t5"] = t5_output[item]
                logTrain_dict['flag'] = '#t5认为错误，api认为错误，t5与api修改不相同'
                logTrain_json[model_json[item]['date']].append(logTrain_dict)
                t5n_apin_d += 1
        elif model_json[item]['tag'] == True and t5_input[item] == t5_output[item]: #api认为错误，t5认为正确
            logTrain_dict["original_text"] = model_json[item]['sentence']
            logTrain_dict["correct_text"] = model_json[item]['fixed']
            logTrain_dict['flag'] = '#api认为错误，t5认为正确'
            logTrain_json[model_json[item]['date']].append(logTrain_dict)
            t5y_apin += 1
        elif model_json[item]['tag'] == False and t5_input[item] != t5_output[item]: #api认为正确，t5认为错误
            logTrain_dict["original_text"] = model_json[item]['sentence']
            logTrain_dict["correct_text"] = t5_output[item]
            logTrain_dict['flag'] = '#api认为正确，t5认为错误'
            logTrain_json[model_json[item]['date']].append(logTrain_dict)
            t5n_apiy += 1
       # print(t5_output)
    result_numdict = {'t5认为正确，api认为正确':t5y_apiy,'t5认为错误，api认为错误，t5与api修改相同':t5n_apin_s,'t5认为错误，api认为错误，t5与api修改不相同':t5n_apin_d,'t5认为正确，api认为错误':t5y_apin,'t5认为错误，api认为正确':t5n_apiy}
    result_numfile = open(result_num,'a',encoding='utf-8')
    result_numfile.write(str(result_numdict))
    for date in logTrain_json.keys():
        logTrain = open(result_path + f'/logTrain_{date}.json', 'a', encoding='utf-8')
        logTrain.write(json.dumps(logTrain_json[date], ensure_ascii=False, indent=2))

if __name__ == '__main__':
    # logpath = 'D:\研究生\pycorrector\log\log_v2_2_cor.log'
    # print(read_LogFile(logpath))
    # log_path = ''
    # right_path = ''
    # wrong_path = ''
    # eval_path = ''
    # log_data = read_LogFile(log_path)

    # right_example = []
    # wrong_example = []

    # tag_dict1=read_LogFile(logpath)
    # model_json=catch_RevisedSentence(tag_dict1)
    # out_json = open('log_eval_catch.json', 'w', encoding='utf-8')
    # out_json.write(json.dumps(model_json, ensure_ascii=False, indent=2))

    # test = '数据分析完成了，接下来我们去现场核查一下污水管网建设、农药包装废弃物回收和重点水库水质情况。'
    # model = T5Corrector('/ddnstor/imu_zhanghuaiwen/mmc_corrector/T5/output/t5_221009_newdata')
    # res = get_t5Pred(model.batch_t5_correct,[test])
    # print(res)

    path = '/home/imu_zhanghuaiwen/ddn/mmc_corrector/T5_corrector/log_v2_07_08_02_cor.log'
    result_path ='/home/imu_zhanghuaiwen/ddn/mmc_corrector/T5_corrector/evalouts/log_v2_07_08_02'

    eval_log(path,result_path)

