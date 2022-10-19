from zhon.hanzi import punctuation
import string
import time
from collections import deque
import re


# 用于对长句进行划分为多个短句
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 注意，分号、破折号、英文双引号等都忽略了，需要的再做些简单调整即可。
    return para.split("\n")

# -------------------------------用于切分长句子为短句子----------------------------------

# 可以调整哪些符号需要切分
def is_cut_zh_mark(c):
    marks = "。；：？！ \n"
    return (c in marks)

def cut_s(s):
    s = re.sub('([""''])', '', s)
    s = re.sub('[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\u0000\u3000\xa0\xa0]+', '', s)
    return s 

def is_delete_mark(c):
    marks = ""
    return(c in marks)

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
    longSentence = cut_s(longSentence)
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

def short2long(cut_text,cut_mark,ori_text):
    res = ''
    for m in cut_mark:
        if isinstance(m,int):
            if abs(len(cut_text[m]) - len(ori_text[m])) <= max(4,int(len(ori_text[m])*0.2)):
                res += cut_text[m]
            else:
                res += ori_text[m]
        else:
            res += m
    return res

# -------------------用于处理模型输出会将符号转化为英文符号的问题，但是现在已经通过后处理的方式给去了，所以好像暂时没用-------------------

# 英文符号转化为中文符号
def E_trans_to_C(string):
    E_pun = u',!?[]()<>:;'
    C_pun = u'，！？【】（）《》：；'
    table= {ord(f):ord(t) for f,t in zip(E_pun,C_pun)}
    return string.translate(table)

def existBlankProblem(sentence):
    if ' ' in sentence:
        return True
    else:
        return False

def fixBlankMiss(correct_text,origin_text):
    c_i = 0
    o_i = 0

    # 用于修正两个指针的对齐状态
    def revise(c_i,o_i):
        save_seat = (c_i,o_i)
        correct_l = len(correct_text) - c_i - 1
        origin_l = len(origin_text) - o_i - 1
        if correct_l + origin_l != 0:
            if correct_l > origin_l:
                c = origin_text[o_i]
                if c in correct_text[c_i:]:
                    c_i = c_i + (correct_text[c_i:].find(c))
                    return (c_i,o_i)
                else:
                    if origin_l:
                        seat = revise(c_i,o_i+1)
                        if seat == (-1,-1):
                            return (save_seat[0]+1,save_seat[1])
                        else:
                            return seat
                    else:
                        return(-1,-1)
            else:
                c = correct_text[o_i]
                if c in origin_text[o_i:]:
                    o_i = origin_text[o_i:].find(c)
                    return (c_i,o_i)
                else:
                    if correct_l:
                        seat = revise(c_i,o_i+1)
                        if seat == (-1,-1):
                            return (save_seat[0],save_seat[1]+1)
                        else:
                            return seat
                    else:
                        return(-1,-1)
        else:
            return save_seat

    while c_i < len(correct_text) and o_i < len(origin_text):
        if correct_text[c_i] == origin_text[o_i]:   #原文与改文一致，不用改变
            c_i += 1
            o_i += 1
        else:   
            if is_blank(origin_text[o_i]):
                correct_text = correct_text[:c_i] + origin_text[o_i] + correct_text[c_i:]
                c_i += 1
                o_i += 1
            else:
                c_i,o_i = revise(c_i,o_i)

    return correct_text

# -------------------------------其他一些辅助的函数----------------------------------------
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

def is_blank(w):
    if w == ' ':
        return True
    else:
        return False