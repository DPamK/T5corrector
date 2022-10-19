# -*- coding:utf-8 -*-
import json
import re
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.fixModeloutput import E_trans_to_C
import torch
from loguru import logger
import time
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_LogFile(logpath):
    '''
    功能：读取log文件，按照以下结构返回一个log信息的list
    logdata = [
        {
            'source': sentence,
            'alerts':alerts
            'tag':True
            'fixed':fixed
        },
        ...
    ]
    sentence ，为输入的语句
    alert，为模型返回信息
    tag, 为该句是否需要修改，有修改为True，无修改为False
    '''
    logfile = open(logpath,encoding='utf-8')
    tag_list = {}
    flag = ''
    for log_line in logfile:
        id = re.search(r'[0-9]{8,15}\.[0-9]+',log_line)
        if id==None:
            continue
        if id.group() not in tag_list.keys():
            tag_list[id.group()] = {}
        if '===>' in log_line:
            sen_id = re.search(r'[0-9]{8,15}\.[0-9]+',log_line)
            sen_source = log_line.split('===>')
            sen_source = json.loads(sen_source[1])
            try:
                tag_list[sen_id.group()]['sentence'] = sen_source['sentences'][0]
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
            if len(tag_list[alert_id.group()]['alerts'][0])==0:
                tag_list[alert_id.group()]['tag'] = False
            else:
                tag_list[alert_id.group()]['tag'] = True
        else:
            continue
    # print(tag_list)
    # out_json = open('today.json','w',encoding='utf-8')
    tag_dict = [i for i in tag_list.values() if len(i)>2]
    # out_json.write(json.dumps(tag_dict, ensure_ascii=False, indent=2))
    return tag_dict

def catch_RevisedSentence(tag_dict1):
    '''
    功能：输入source,alerts，返回纠正后的句子fixed
    '''
    model_json=[]
    for tag_dict in tag_dict1:
        # print(tag_dicts)
        source=tag_dict['sentence']
        alerts=tag_dict['alerts']
        tag=tag_dict['tag']
    # for source,alerts,tag in zip(tag_dict['sentence'],tag_dict['alerts'],str(tag_dict['tag'])):
        if not tag:#不修改
            tag_dict['fixed']=source
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
                if item['alertType']==4 or item['alertType']==3 or item['alertType']==16:#替换
                    ori_content=source[start:end+1]
                    source=source[:start]+replaceText+source[end+1:]
                    change_len=len(replaceText)-len(ori_content)
                elif item['alertType']==1:#向前新增
                    source = source[:start] + replaceText + source[start:]
                    change_len = len(replaceText)
                elif item['alertType']==2:#向后新增
                    source = source[:end + 1] + replaceText + source[end+1:]
                    change_len = len(replaceText)
                if change_len!=0:
                    change(change_len, alerts[0], end)
            tag_dict['alerts']=ori_alert
            tag_dict['fixed']=source
            model_json.append(tag_dict)
    return model_json

def change(change_len,alert,index):
    for item in alert:
        start=item["start"]
        end=item["end"]
        if start>=index:
            item['start']+=change_len
            item['end']+=change_len

class T5Corrector(object):
    def __init__(self,model_dir='t5_best'):
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
            decode_tokens = self.tokenizer.decode(outputs[i]).replace('<pad>', '').replace('</s>', '').replace(' ', '').replace('<unk>', '')
            corrected_text = decode_tokens
            corrected_text = get_errors(corrected_text, text)
            text_new += corrected_text
            # sub_details = [(i[0], i[1], idx + i[2], idx + i[3]) for i in sub_details]
            # details.extend(sub_details)
            result.append(text_new)
        return result

def get_t5Pred(correct_fn,sources):
    res = correct_fn(sources)
    return res

def get_errors(corrected_text, origin_text):
    corrected_text = E_trans_to_C(corrected_text)
    return corrected_text



if __name__== '__main__':
    logpath = 'D:\研究生\pycorrector\log\log_v2_2_cor.log'
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

    test = '数据分析完成了，接下来我们去现场核查一下污水管网建设、农药包装废弃物回收和重点水库水质情况。'
    model = T5Corrector('save_models/t5_newdata_1009')
    res = get_t5Pred(model.batch_t5_correct,[test])
    print(res)




