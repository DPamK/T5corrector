# -*- coding:utf-8 -*-
from cgitb import text
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import math
from loguru import logger
import time
import json
import re
from utils.fixModeloutput import E_trans_to_C,existBlankProblem,fixBlankMiss,long2short,short2long
import pandas as pd
import requests
from transfer import post_process_sequence,deal_with_alert
import jieba
from zhon.hanzi import punctuation
import string

os.environ["CUDA_VISIBLE_DEVICES"]='6,7'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']
model_dir = "t5_ad_model"

# 用于对长句进行划分为多个短句
def cut_sent(para):
    para = re.sub('([。！？\? \n\t])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 注意，分号、破折号、英文双引号等都忽略了，需要的再做些简单调整即可。
    return para.split("\n")

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

def saveJson(data,filePath):
    json_dicts = json.dumps(data,indent=4,ensure_ascii=False)
    with open(filePath,'w+',encoding='utf8') as fp:
        fp.write(json_dicts)


def cut_content(content):
    sentences = cut_sent(content)
    sentences_index = []
    start = 0
    for text in sentences:
        end = len(text) + start
        temp = (start,end-1)
        sentences_index.append(temp)
        start = end
    return sentences,sentences_index

def run_content(file_path,res_path):
    
    files = os.listdir(file_path)
    for file in files:
        if file == 'error.json':
            continue
        path = os.path.join(file_path,file)
        print(path)
        dir_path = os.path.join(res_path,file.split('.')[0])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(path,'r',encoding='utf8') as fp:
            alljson = json.load(fp)
        num = 0
        for item in alljson:
            id = item['id']
            site_url = item['site_url']
            site_title = item['site_title']
            site_content = item['site_content']
            errors = item['detected_errors']['detected_errors'][0]
            name = id  +'.json'
            print(f'{num}:{id}--{site_title}') 
            num += 1
            sentences,index = cut_content(site_content)
            res = []
            for sid,sentence in enumerate(sentences):
                start,end = index[sid]
                elist = []
                for error in errors:
                    if start <= error['start_index'] <= end:
                        error['start_index'] = error['start_index'] - start
                        error['end_index'] = error['end_index'] - start
                        elist.append(error)
                temp = {
                    'sentence_id':sid,
                    'sentence':sentence,
                    'detected_errors':elist
                }
                res.append(temp)
            whole = {
                'id':id,
                'site_url':site_url,
                'site_title':site_title,
                'output':res
            }
            
            saveJson(whole,os.path.join(dir_path,name))

def analyse_dir_errors(dir_path,res_path):
    add_front = []
    add_after = []
    delete = []
    replace = []
    files = os.listdir(dir_path)
    num = 0
    for file in files:
        path = os.path.join(dir_path,file)
        print(path)
        with open(path,'r',encoding='utf8') as fp:
            alljson = json.load(fp)
        
        id = alljson['id']
        site_url = alljson['site_url']
        site_title = alljson['site_title']
        output = alljson['output']
        print(f'{num}:{id}--{site_title}') 
        num += 1
        for item in output:
            sentence = item['sentence']
            errors = item['detected_errors']
            if len(errors) == 0:
                continue
            for error in errors:
                sug = error['suggestion']
                source = error['source']
                fixed = error['fixed']
                smallitem = {
                    'site_id':id,
                    'sentence':sentence,
                    'source':source,
                    'fixed':fixed,
                    'start_index':error['start_index'],
                    'end_index':error['end_index']
                }
                if sug == '向前新增':
                    add_front.append(smallitem)
                elif sug == '向后新增':
                    add_after.append(smallitem)
                elif sug == '删除':
                    delete.append(smallitem)
                elif sug == '替换':
                    replace.append(smallitem)
    res = {
        'add_front':add_front,
        'add_after':add_after,
        'delete':delete,
        'replace':replace
    }
    saveJson(res,res_path)

def run_analyse_divide(files_path,res_path):
    files = os.listdir(files_path)
    for file in files:
        print(f'analyse {file}')
        analyse_dir_errors(os.path.join(files_path,file),os.path.join(res_path,file+'.json'))


def run_list_content(file_path,res_path):
    dirs = os.listdir(file_path)
    add_front = []
    add_after = []
    delete = []
    replace = []
    lenth = 0
    for dir in dirs:
        files = os.listdir(os.path.join(file_path,dir))
        num = 0
        for file in files:
            path = os.path.join(file_path,dir,file)
            print(path)
            with open(path,'r',encoding='utf8') as fp:
                alljson = json.load(fp)
            
            id = alljson['id']
            site_url = alljson['site_url']
            site_title = alljson['site_title']
            output = alljson['output']
            print(f'{num}:{id}--{site_title}') 
            num += 1
            for item in output:
                
                sentence = item['sentence']
                # lenth += len(item['sentence'])
                errors = item['detected_errors']
                if len(errors) == 0:
                    continue
                else:
                    lenth += len(item['sentence'])
                for error in errors:
                    sug = error['suggestion']
                    source = error['source']
                    fixed = error['fixed']
                    advice = error['advise']
                    smallitem = {
                        'site_id':id,
                        'dir':dir,
                        'sentence':sentence,
                        'suggestion':sug,
                        'advice':advice,
                        'source':source,
                        'fixed':fixed,
                        'start_index':error['start_index'],
                        'end_index':error['end_index']
                    }
                    if sug == '向前新增':
                        add_front.append(smallitem)
                    elif sug == '向后新增':
                        add_after.append(smallitem)
                    elif sug == '删除':
                        delete.append(smallitem)
                    elif sug == '替换':
                        replace.append(smallitem)
    saveJson(add_front,os.path.join(res_path,'add_front.json'))
    saveJson(add_after,os.path.join(res_path,'add_after.json'))
    saveJson(delete,os.path.join(res_path,'delete.json')) 
    saveJson(replace,os.path.join(res_path,'replace.json'))
    print(f'总字数：{lenth}')

def exist_digits(text):
    for c in text:
        if c.isdigit():
            return True
    return False

def supple_error(errors,sentence):
    textlist = jieba.lcut(sentence)
    textlenth = [len(text) for text in textlist]
    res = []
    hash = []
    for error in errors:
        if error['suggestion'] == '谨慎查验':
            res.append(error)
        else:
            start = error['start_index']
            end = error['end_index'] + 1
            ori = error['source']
            fix = error['fixed']
            if len(ori) == len(fix) and len(ori)==1:
                if is_zh_punctuation(fix) and is_en_punctuation(ori):
                    res.append(error)
                    continue
            if exist_digits(ori):
                res.append(error)
                continue
            num = 0
            i = 0
            while num + textlenth[i] < start:
                    num += textlenth[i]
                    i += 1
            
            error['start_index'] = num
            
            j = i
            while num + textlenth[j] < end:
                num += textlenth[j]
                j += 1
            source = ''
            if j+2 >= len(textlist):
                for t in range(i-1,j+1):
                    source += textlist[t]
                error['source'] = source
                error['end_index'] = num + textlenth[j]
                error['start_index'] = num - textlenth[i]
            else:
                for t in range(i-1,j+2):
                    source += textlist[t]
                error['source'] = source
                error['end_index'] = num + textlenth[j] + textlenth[j+1]
                error['start_index'] = num - textlenth[i]

            if error['suggestion'] == '替换':
                x = source.replace(ori,fix)
            elif error['suggestion'] == '删除':
                x = source.replace(ori,'')
            elif error['suggestion'] == '向前新增':
                x = source.replace(ori,fix+ori)
            elif error['suggestion'] == '向后新增':
                x = source.replace(ori,ori+fix)
            else:
                x = source.replace(ori,fix)
            error['fixed'] = x
            sug = error['suggestion']
            fix = error['fixed']
            sou = error['source']
            item = 'suggest:{0}\t\tsource:{1}\t\tfixed:{2}\n'.format(sug,sou,fix)
            res.append(error)
            hash.append(item)
    return res,hash

def alert2txt(advice,savepath,key_lenth=4,appearnum=6):
    res = []
    for key in advice:
        if len(key)>key_lenth:
            if advice[key] > appearnum:
                res.append(key)
    with open(savepath,'w+',encoding='utf8') as f:
        for item in res:
            f.write(item)
            f.write('\n')

def run_frequncefn(dirpath,save_path,needblanktxt=False,blanktxtpath='blank_words/addpairs_2.txt',needAdviceTxt=False):
    freq_only_path = 'freq_error.json'
    save_path = os.path.join(save_path,freq_only_path)
    adviceReplace = {}
    adviceInsert = {}
    adviceDelete = {}
    hash = {}
    text_hash = {}
    dirs = os.listdir(dirpath)
    for dir in dirs:
        files = os.listdir(os.path.join(dirpath,dir))
        for file in files:
            path = os.path.join(dirpath,dir,file)
            print(path)
            with open(path,'r',encoding='utf8') as fp:
                alljson = json.load(fp)
            id = alljson['id']
            print(id)
            output = alljson['output']
            for item in output:
                sentence = item["sentence"]
                errors = item['detected_errors']
                for error in errors:
                    sug = error['suggestion']
                    if sug == '谨慎查验':
                        continue
                    else:
                        sug = error['suggestion']
                        sou = error['source']
                        fix = error['fixed']
                        advise = error['advise']
                        if advise == 'False':
                            text = f'{sug}:{sou}->{fix}'
                            if text in hash:
                                hash[text] += 1
                                if len(text_hash[text]) < 6:
                                    text_hash[text].append(sentence)
                            else:
                                hash[text] = 1
                                text_hash[text] = [sentence]
                        elif advise == 'True':
                            if sug == '替换':
                                text = f'{sou}->{fix}'
                                if text in adviceReplace:
                                    adviceReplace[text] += 1
                                else:
                                    adviceReplace[text] = 1
                            elif sug == '向后新增':
                                text = f'{sou}->{fix}'
                                if text in adviceInsert:
                                    adviceInsert[text] += 1
                                else:
                                    adviceInsert[text] = 1
                            elif sug == '删除':
                                text = sou
                                if text in adviceDelete:
                                    adviceDelete[text] += 1
                                else:
                                    adviceDelete[text] = 1                                 
            

    if needAdviceTxt:
        alert2txt(adviceReplace,'blank_words/replace_2.txt')
        alert2txt(adviceInsert,'blank_words/insert_3.txt')
        alert2txt(adviceDelete,'blank_words/delete_4.txt',key_lenth=1,appearnum=3)
    
    hash_sort = sorted(hash.items(),key = lambda x:x[1],reverse = True)
    new_hash = [[change,freq,text_hash[change]] for change,freq in hash_sort]
    saveJson(new_hash,save_path)


    if needblanktxt:
        pairs = [key for key,value in hash.items() if value > 6 and len(key) > 4]
        with open(blanktxtpath,'w+',encoding='utf8') as f:
            for i in pairs:
                f.write(i)
                f.write('\n')

if __name__ == '__main__':

    # 进行扩展
    file_path = 'outputLargeData/区县数据'
    short_path = 'smalloutLargeData/其他行业数据'
    res_path = 'largeallerror.json'
    hash_path = 'largesmallerror.json'
    freq_only_path = 'freq_error.json'
    blank_pair_path = 'addpairs_2.txt'
    analyse_path = 'analyse/其他行业数据'
    

    # run_content(file_path,short_path)
    run_list_content(short_path,analyse_path)
    run_frequncefn(short_path,analyse_path,needAdviceTxt=False)
    
    # res = []
    # hash = {}
    # files = os.listdir(file_path)
    # for file in files:
    #     if file == 'error.json':
    #         continue
    #     path = os.path.join(file_path,file)
    #     print(path)
    #     with open(path,'r',encoding='utf8') as fp:
    #         alljson = json.load(fp)
    #     for item in alljson:
    #         id = item['id']
    #         print(id)
    #         site_url = item['site_url']
    #         site_title = item['site_title']
    #         site_content = item['site_content']
    #         errors = item['detected_errors']['detected_errors'][0]
    #         for error in errors:
    #             sug = error['suggestion']
    #             if sug == '谨慎查验':
    #                 continue
    #             else:
    #                 sug = error['suggestion']
    #                 sou = error['source']
    #                 fix = error['fixed']
    #                 advise = error['advise']
    #                 # if advise == 'False':
    #                 text = f'{sug}:{sou}->{fix}'
    #                 if text in hash:
    #                     hash[text] += 1
                        
    #                 else:
    #                     hash[text] = 1
                    
    #                 res.append(error)
            
    # # hash = list(set(hash))
    
    # pairs = [key for key,value in hash.items() if value > 6 and len(key) > 4]
    # hash_sort = sorted(hash.items(),key = lambda x:x[1],reverse = True)
    # saveJson(hash_sort,freq_only_path)
    # saveJson(res,res_path)

    # # with open(blank_pair_path,'w+',encoding='utf8') as f:
    # #     for i in pairs:
    # #         f.write(i)
    # #         f.write('\n')


   
    
            
        