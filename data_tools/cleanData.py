# -*- coding:utf-8 -*-
import difflib
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import math
from loguru import logger
from zhon.hanzi import punctuation
import time
import json
from utils.fixModeloutput import E_trans_to_C
import pandas as pd
from collections import deque
from utils.dictfilter import dictfiliter
import string
from utils.fixModeloutput import short2long
import pycorrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------前置设置部分----------------------------------------
# model_dir = "output/t5_huge_nosymbol"
model_dir = "shibing624/mengzi-t5-base-chinese-correction"


# --------------------------------------------函数部分----------------------------------------

# 针对解决模型输出结果符号变为英文符号的问题
def get_errors(corrected_text, origin_text):
    corrected_text = E_trans_to_C(corrected_text) 
    return corrected_text

def get_t5Pred(correct_fn,sources):
    res = correct_fn(sources)
    return res

# 模型
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

#加载模型
model = T5Corrector(model_dir)

#保存json
def saveJson(data,filePath):
    json_dicts = json.dumps(data,indent=4,ensure_ascii=False)
    with open(filePath,'w+',encoding='utf8') as fp:
        fp.write(json_dicts)

def is_cut_zh_mark(c):
    marks = "。；：？！"
    return (c in marks)

def is_Chinese(w):
    return '\u4e00' <= w <= '\u9fff'

def is_zh_punctuation(w):
    punctuation_str = punctuation   #中文符号
    return w in punctuation_str

def is_en(w):
    if 'a'<=w<='z' or 'A'<=w<='Z':
        return True
    else:
        return False

def is_en_punctuation(w):
    return w in string.punctuation

def is_sentence(text):
    zh_num = 0
    zh_mark = False
    for char in text:
        if is_Chinese(char):
            zh_num += 1
        if is_cut_zh_mark(char):
            zh_mark = True
    if zh_num < len(text)*0.7:
        return False
    elif zh_num < 2:
        return False
    elif is_cut_zh_mark(text[0]):
        return False
    elif text[-1].isdigit():
        return False
    else:
        return zh_mark

def needchar(char):
    if is_Chinese(char) or is_zh_punctuation(char):
        return True
    elif char.isdigit():
        return True
    elif is_en_punctuation(char):
        if char in "[]()":
            return True
    else:
        return False

def is_cut_zh_mark(c):
    marks = "，。；：？！"
    if needchar(c):
        return (c in marks)
    else:
        return True

# 可以调整那些可能需要切缝的符号（需要两个的符号）
def is_cut_judge_mark(c):
    marks = "“”（）《》‘’"
    return (c in marks)
def match_cpMark(mark1,mark2):
    if mark1 == "“" and mark2 == "”":
        return True
    elif mark1 == "《" and mark2 == "》":
        return True
    elif mark1 == "（" and mark2 == "）":
        return True
    elif mark1 == "‘" and mark2 == "’":
        return True
    else:
        return False

def long2short(longSentence):
    lenth = len(longSentence)
    cut_text = []   #保存文字
    cut_mark = []   #保存符号
    stark = deque() #保存配对符号
    i = 0           #开始
    j = 0           #指针
    num = 0         #文字编号
    while j < lenth:
        if is_cut_zh_mark(longSentence[j]):    #判断是否是需要切缝的符号
            if i != j:                         #如果是，判断是否有字符需要保存
                #需要保存就保存
                cut_text.append(longSentence[i:j])
                cut_mark.append(num)
                num += 1

            cut_mark.append(longSentence[j])

            j += 1
            i = j
        else:
            if is_cut_judge_mark(longSentence[j]):  #判断是否是需要配对的符号
                if j+1 == lenth:                    #保证这个配对符不在末尾
                    j = lenth
                    continue
                need_cut = False                    #用于判断配对符号是否需要再切分
                stark.append(longSentence[j])
                temp = j+1
                while temp < lenth:
                    if is_cut_judge_mark(longSentence[temp]):
                        left_mark = stark.pop()
                        if not match_cpMark(left_mark,longSentence[temp]):
                            stark.append(left_mark)
                            stark.append(longSentence[temp])
                            temp += 1
                        else:
                            if not stark:
                                if need_cut:
                                    cut_text.append(longSentence[i:j])
                                    cut_mark.append(num)
                                    num += 1

                                    tempstr = longSentence[j+1:temp]
                                    temp_text,temp_mark = long2short(tempstr)
                                    cut_mark.append(longSentence[j])
                                    for m in temp_mark:
                                        if isinstance(m,int):
                                            cut_text.append(temp_text[m])
                                            cut_mark.append(num)
                                            num += 1
                                        else:
                                            cut_mark.append(m)
                                    cut_mark.append(longSentence[temp])
                                    i = temp + 1
                                    j = i
                                else:
                                    temp += 1
                                    j = temp
                                break
                            else:
                                need_cut = True
                                temp += 1

                    elif is_cut_zh_mark(longSentence[temp]):
                        need_cut = True
                        temp += 1
                    else:
                        temp += 1
                        if temp == lenth:
                            if need_cut:
                                j += 1
                            else:
                                j = lenth
                # temp走到了最后，且stark没有清空，配对符不匹配

                if  j != lenth and stark:
                    if need_cut:
                        if i != j:
                            cut_text.append(longSentence[i:j])
                            cut_mark.append(num)
                            num += 1
                        cut_mark.append(longSentence[j])
                        temp_text,temp_mark = long2short(longSentence[j+1:])
                        for m in temp_mark:
                            if isinstance(m,int):
                                cut_text.append(temp_text[m])
                                cut_mark.append(num)
                                num += 1
                            else:
                                cut_mark.append(m)
                        j = lenth
                        i = j

                    else:
                        cut_text.append(longSentence[i:temp])
                        cut_mark.append(num)
                        j = lenth
                        i = j
            else:
                j += 1
    if i != j:
        cut_text.append(longSentence[i:j])
        cut_mark.append(num)

    return cut_text,cut_mark

def catchSentence(text):
    '''
    input: 一篇文章、一段长文字
    output: 从文章中提取到的完整的句子组成的一个list，完整句子在list的id与其他部分组成的list
    '''
    sentences = []
    other_part = []
    text,mark = long2short(text)
    num = 0 
    temp = ''
    stack = deque()
    for item in mark:
        if isinstance(item,int):
            temp += text[item]
        elif is_cut_judge_mark(item):
            temp += item
            if len(stack) != 0:
                left = stack.pop()
                if not match_cpMark(left,item):
                    stack.append(left)
                    stack.append(item)
                else:
                    if len(stack) == 0:
                        if is_sentence(temp):
                            sentences.append(temp)
                            other_part.append(num)
                            num += 1
                            temp = ''
            else:
                stack.append(item)
        else:
            if needchar(item):
                temp += item
                if is_cut_zh_mark(item) and item != '，':
                    if len(stack) == 0:
                        if is_sentence(temp):
                            sentences.append(temp)
                            other_part.append(num)
                            num += 1
                            temp = ''
                        else:
                            other_part.append(temp)
                            temp = ''
            else:
                if is_sentence(temp):
                    if temp != '':
                        sentences.append(temp)
                        other_part.append(num)
                        num += 1
                        temp = ''
                    
                    other_part.append(item)
                else:
                    temp += item
                    other_part.append(temp)
                    temp = ''
    if temp != '':
        if is_sentence(temp):
            sentences.append(temp)
            other_part.append(num)
            num += 1
        else:
            other_part.append(temp)

    return sentences,other_part

def list_of_groups(list_info, per_list_len):
    '''
    :param list_info:   列表
    :param per_list_len:  每个小列表的长度
    :return:
    '''
    list_of_group = zip(*(iter(list_info),) *per_list_len) 
    end_list = [list(i) for i in list_of_group] # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count !=0 else end_list
    return end_list

def inferList(alltext):
    short_text = []
    sentence_mark = []
    biases = []
    num = 0
    for text in alltext: 
        cut_text,cut_mark = long2short(text)
        biases.append(num)
        num += len(cut_text)
        short_text.extend(cut_text)
        sentence_mark.append(cut_mark)

    if len(short_text):
        if len(short_text) < 256:
            short_res = get_t5Pred(model.batch_t5_correct,short_text)
        else:
            short_res = []
            max_l = 200
            num = math.ceil(len(short_text)/max_l)
            for i in range(num):
                if i+1 == num:
                    temp = short_text[i*max_l:]
                else:
                    temp = short_text[i*max_l:(i+1)*max_l]
                tempp = get_t5Pred(model.batch_t5_correct,temp)
                short_res.extend(tempp)
    else:
        short_res = short_text
    
    predlist = short_res
    output = []
    for marks,bias in zip(sentence_mark,biases):
        temp = ''
        for mark in marks:
            if isinstance(mark,int):
                temp += predlist[mark+bias]
            else:
                temp += mark
        output.append(temp)
    
    return output

def cleanSentence(filepath,output):
    longSentences = []
    with open(filepath, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            newstr = line.rstrip('\n')
            longSentences.append(newstr)
    logger.info('读取长句')
    shortSentences = []
    for longSentence in longSentences:
        item,other_part = catchSentence(longSentence)
        for shortSentence in item:
            shortSentences.append(shortSentence)
    logger.info('切分短句')

    if not os.path.exists(output):
        os.makedirs(output)

    f_correct = open(os.path.join(output,'correct.txt'), 'w+', encoding="utf-8")
    f_incorrect = open(os.path.join(output,'incorrect.txt'), 'w', encoding="utf-8")
    batch_num = 300
    right_num = 0
    wrong_num = 0
    right_word = 0
    wrong_word = 0
    
    num = math.ceil(len(shortSentences)/batch_num)
    for i in range(num):
        if i+1 == num:
            temp = shortSentences[i*batch_num:]
        else:
            temp = shortSentences[i*batch_num:(i+1)*batch_num]
        tempp = inferList(temp)
        for source,pred in zip(temp,tempp):
            if source==pred:
                f_correct.write(str(source) + '\n')
                right_num += 1
                right_word += len(source)
            else:
                f_incorrect.write(str(source) + '\n')
                wrong_num += 1
                wrong_word += len(source)
        logger.info(f'已处理的batch：{i}')
    logger.info(f'正确的总句数：{right_num}，错误的总句数：{wrong_num}，正确的总字数：{right_word}，错误的总字数：{wrong_word}，')
    f_correct.close()
    f_incorrect.close()


if __name__ == '__main__':
    
    file_path = 'motherdata/result/correct.txt'
    output = 'motherdata/result/origin'

    cleanSentence(file_path,output)

   