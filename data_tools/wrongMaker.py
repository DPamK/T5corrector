import json
import random
from LAC import LAC
from pypinyin import pinyin,Style
import os
import math
from multiprocessing import pool

notchoose_tag=['PER','LOC','TIME','m','w','nz','nw','xc']


class Condition:
    def judge_condition(self,change_num,priority,tag,random_run):
        flag=[0,1]
        if change_num<2:#控制修改次数
            if priority<3 and tag not in notchoose_tag:#控制修改词语条件
                if random_run==True:#控制随机执行
                    if random.choice(flag)==1:#随机决定此处是否修改（确保不是按顺序修改）
                        return 'change'
                    else:
                        return 'e_change'
                else:
                    if random.choice(flag)==1:
                        return 'e_random_change'
                    else:
                        return 'e_random_e_change'
            else:
                return 'e_condition'
        else:
            return 'e_num'
        

def read_text(path):
    file = open(path, encoding='utf-8')
    sentences_list = []
    for line in file:
        line = line.strip('\n')
        if line == '':
            continue
        if len(line) < 128:
            sentences_list.append(line)
    return sentences_list


def split_sentence(full_list, ratios, shuffle=True):
    random.seed(2)  # 随机种子
    n_total = len(full_list)
    offset = [0] * len(ratios)  # 存储数量
    for index in range(len(ratios)):
        offset[index] = int(n_total * ratios[index])
        if index >= 1:
            offset[index] = offset[index - 1] + offset[index]
        if n_total == 0 or offset[index] < 1:
            return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist = [None] * len(ratios)
    sublist[0] = full_list[:offset[0]]
    
    for index in range(1, len(ratios)):
        sublist[index] = full_list[offset[index - 1]:offset[index]]
    return sublist

lac = LAC(mode='rank')

def load_importence(texts):
    
    rank_result = lac.run(texts)
    return rank_result

#正确的
def correct(sublist):
    content_list=[]
    for item in sublist:
        pair={}
        pair['original_text']=item
        pair['correct_text']=item
        content_list.append(pair)
    return content_list
    
#少字
def delete_word(item):
    change_num=0#控制一句话中的修改数量小于2
    word_after_change=[None]*len(item[0])
    word=item[0]
    tag=item[1]
    priority=item[2]

    for index in range(len(word)):#随机选取一个数
        c = Condition()
        result=c.judge_condition(change_num,priority[index],tag[index],True)
        if result=='change':
            word_after_change[index] = word[index].replace(random.choice(list(word[index])), "")
            change_num += 1
        elif result=='e_num':
            word_after_change[index:] = word[index:]
            break
        else:
            word_after_change[index] = word[index]
   
    return get_content_list(word,word_after_change)

#多字
def add_word(item):
    change_num = 0  # 控制一句话中的修改数量小于等于2
    word_after_change = [None] * len(item[0])
    word = item[0]
    tag = item[1]
    priority = item[2]
    for index in range(len(word)):  # 随机选取一个数
        c = Condition()
        result = c.judge_condition(change_num, priority[index], tag[index], True)
        if result == 'change':
            word_after_change[index] = word[index] + word[index]
            change_num += 1
        elif result == 'e_num':
            word_after_change[index:] = word[index:]
            break
        else:
            word_after_change[index] = word[index]
    return get_content_list(word,word_after_change)

#词序颠倒
def reverse_word(item):
    change_num = 0  #
    word_after_change = [None] * len(item[0])
    word = item[0]
    tag = item[1]
    priority = item[2]
    for index in range(0,len(word) - 1,2):  # 随机选取一个数
        c = Condition()
        result = c.judge_condition(change_num, priority[index], tag[index], True)
        if result == 'change':
            if priority[index + 1] < 3 and tag[index + 1] not in notchoose_tag:  # 前后两个词都符合
                word_after_change[index] = word[index + 1]
                word_after_change[index + 1] = word[index]
                change_num+=1
                continue
            # else:
            #     word_after_change[index] = word[index]
            #     word_after_change[index + 1] = word[index]
        elif result == 'e_num':
            word_after_change[index:] = word[index:]
            break
        # else:
        word_after_change[index] = word[index]
        word_after_change[index + 1] = word[index+1]
    word_after_change[-1]=word[-1]
    if len(word)==1:
        word_after_change[0]=word[0]
    return get_content_list(word,word_after_change)

#出现错别字
def wrong_word(item,xingsi_table,yinsi_table):#一句话出现两个错字
    change_num = 0  # 控制一句话中的修改数量小于等于2
    word_after_change = [None] * len(item[0])
    word = item[0]
    tag = item[1]
    priority = item[2]
    for index in range(len(word)):
        c = Condition()
        random_run=random_unit(0.5)
        result = c.judge_condition(change_num, priority[index], tag[index], random_run)
        ori_word = random.choice(list(word[index]))
        if result == 'change':#形似
            xingsi_word = similarword(ori_word, xingsi_table)
            if  xingsi_word!=None:
                # print(f"形似字：{xingsi_word}")
                word_after_change[index] = word[index].replace(ori_word, xingsi_word)
                change_num += 1
                continue
        elif result=='e_random_change':#音似
            yinsi_word=get_heteronym_char(ori_word,yinsi_table)
            if yinsi_word!=None:
                word_after_change[index] = word[index].replace(ori_word, yinsi_word)
                change_num += 1
                continue
        elif result == 'e_num':
            word_after_change[index:] = word[index:]
            break
        word_after_change[index] = word[index]
    return get_content_list(word,word_after_change)

def get_SimilarWord(similar_path):
    file=open(similar_path,encoding="utf-8")
    res = []
    for line in file:
        line = line.strip('\n')
        res.append(line)
    return res

def similarword(ori_word,similar_table):
    file=similar_table
    for line in file:
        if ori_word in line:
            xingsi_list=list(line)
            xingsi_list.remove(ori_word)
            xingsi_word=random.choice(xingsi_list)
            return xingsi_word
    return None


def get_all_char_pinyin(path):
    pinyin_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            ch = line.strip()
            ch_pinyin = pinyin(ch, style=Style.TONE3, heteronym=True)
            # heteronym 是否启用多音字模式
            for p_li in ch_pinyin:
                for p in p_li:
                    if p not in pinyin_dict:
                        pinyin_dict[p] = [ch]
                    else:
                        pinyin_dict[p].append(ch)
    return pinyin_dict


# 获得同音字
def get_heteronym_char(c,yinsi_table, homo=False):
    pinyin_dict = yinsi_table
    cn_pinyin = pinyin(c, style=Style.TONE3, heteronym=homo)
    res = []
    for p_li in cn_pinyin:
        for p in p_li:
            if p in pinyin_dict:
                res.extend(pinyin_dict[p])
            else:
                return c
    if c in res:
        res.remove(c)
    if len(res) == 0:
        a = c
    else:
        a = random.choice(res)
        # print(f"a:{a}")
    return a

def random_unit(p: float):
    if p == 0:
        return False
    if p == 1:
        return True

    R = random.random()
    if R < p:
        return True
    else:
        return False


def get_content_list(word,word_after_change):
    content_list=[]
    ori_sentence, cha_sentence = "", ""
    pair = {}
    for ori_item, cha_item in zip(word, word_after_change):
        if cha_item == None:
            print(word)
            print(word_after_change)
        ori_sentence += ori_item
        cha_sentence += cha_item
    pair['original_text'] = cha_sentence.rstrip('\n')
    pair['correct_text'] = ori_sentence.rstrip('\n')
    content_list.append(pair)
    return content_list

# -------------------------------用于切分长句子为短句子----------------------------------
import re
from collections import deque

# 可以调整哪些符号需要切分
def is_cut_zh_mark(c):
    marks = "，。；：？！ \n"
    return (c in marks)

def cut_s(s):
    s = re.sub('([""''])', '', s)
    s = re.sub('[　  \001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\u0000]+', '', s)
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

def is_Chinese(w):
    return '\u4e00' <= w <= '\u9fff'

def is_sentence(text):
    
    if len(text)==0:
        return False
    else:
        return True


if __name__ == '__main__':
    path=r"motherData/txtdata"
    xingsi_path=r"similar.txt"
    yinsi_path=r"all_3500_chars.txt"
    
    test_path=r"datas/test/test.json"
    valid_path=r'datas/test/logright.json'
    train_path=r'datas/test/train.json'
    sentences_list=[]
    xingsi_table = get_SimilarWord(xingsi_path)
    yinsi_table = get_all_char_pinyin(yinsi_path)

    files = os.listdir(path)
    for file in files:
        temp = read_text(os.path.join(path,file))
        times = int(file.split('.')[0].split('_')[-1])
        for i in range(times):
            sentences_list.extend(temp)
        l = len(temp)*times
        print(f'{file} num is {l}')
        
    l = len(sentences_list)
    print(f'all sentences is {l}')

    # midlist = sentences_list
    # sentences_list = []
    # for text in midlist:
    #     cut_text,_ = long2short(text)
    #     for ctone in cut_text:
    #         if is_sentence(ctone):
    #             sentences_list.append(ctone)
    # l = len(sentences_list)
    # print(f'all sentences has been cutted is {l} ')

    #划分数据：正确，错别字，少字，多字，语序错误
    sublist=split_sentence(sentences_list,[0.6,0.18,0.1,0.1,0.02])
    # print(sublist[3])
    # print(len(sublist))
    #获取命名实体,进行相应的操作
    total = []
    correct_list = correct(sublist[0])
    total+=correct_list
    for index in range(1,len(sublist)):
        templist = sublist[index]
        max_l = 20000
        rank_results=[]
        num = math.ceil(len(templist)/max_l)
        print(f'第{index}类情况 需要{num}批次 paddlelac完成')
        for i in range(num):
            if i+1 == num:
                temp = templist[i*max_l:]
            else:
                temp = templist[i*max_l:(i+1)*max_l]
            tempp = load_importence(temp)
            rank_results.extend(tempp)
            print(f'第{index}类情况 第{i}批次 paddlelac完成')
        l = len(templist)
        print(f'第{index}类情况 共有{l}条')
        
        try:
            for rank_result in rank_results:
                print(rank_result)
                if index==1:
                    wrong_list=wrong_word(rank_result,xingsi_table=xingsi_table,yinsi_table=yinsi_table)
                    total+=wrong_list
                    continue
                elif index==2:
                    delete_list=delete_word(rank_result)
                    total+=delete_list
                    continue
                elif index==3:
                    add_list=add_word(rank_result)
                    total+=add_list
                    continue
                elif index==4:
                    reverse_list=reverse_word(rank_result)
                    total+=reverse_list
                    continue
        except Exception as e:
            raise e
    # print(total)
    random.shuffle(total)
    #划分数据集  0.1 0.1 0.8
    total_len=len(total)
    test=total[:20000]
    valid=total[20000:40000]
    train=total[40000:]
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(test, ensure_ascii=False, indent=2))
    with open(valid_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(valid, ensure_ascii=False, indent=2))
    with open(train_path,'w', encoding='utf-8') as f:
        f.write(json.dumps(train, ensure_ascii=False, indent=2))
