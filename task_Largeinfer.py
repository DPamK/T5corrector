# -*- coding:utf-8 -*-
import difflib
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import math
from loguru import logger
import time
import json
from utils.json_creator import creat_alert_item
from utils.filter import first_filter,second_filter
from utils.fixModeloutput import E_trans_to_C,long2short,short2long
import pandas as pd
from ltp import LTP
from utils.transfer import post_process_sequence,deal_with_alert
from utils.analyseOutput import run_content,run_analyse_divide,run_list_content,run_frequncefn
import sqlite3
import unicodedata
import re
from utils.dictfilter import dictfiliter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------前置设置部分----------------------------------------

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']
db_dir = 'database/inner_correct.db'
model_dir = "output/t5_newdata"
logger.add("log/t5_infer.log")
need_model = True

# --------------------------------------------函数部分----------------------------------------
# 将db中的一些词添加到分词库里
def run_db2dict_LTP(dbdir):
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

# 针对解决模型输出结果符号变为英文符号的问题
def get_errors(corrected_text, origin_text):
    corrected_text = E_trans_to_C(corrected_text)
    
    return corrected_text

def get_t5Pred(correct_fn,sources):
    res = correct_fn(sources)
    return res

# create_json_ltp的子功能，将list转换为str
def list2str(strlist):
    res = ''
    for str in strlist:
        res += str
    return res

# 基于ltp分词的compare 实现生成json的任务 
def create_json_ltp(item):
    source = item['originText']
    output = item['output']
    source = ltp_filter(source)
    textlenth = [len(text) for text in source]
    output = ltp_filter(output)
    s = difflib.SequenceMatcher(None,source,output)
    model_json = []
    for (type, ori_pos_begin, ori_pos_end, out_pos_begin, out_pos_end) in s.get_opcodes():
        if type == 'equal':
            continue
        else:            
            sourcelist = source[ori_pos_begin: ori_pos_end]
            sourceText = list2str(sourcelist)
            replacelist = output[out_pos_begin: out_pos_end]
            replaceText = list2str(replacelist)
            alertMessage = ""
            alertType,errorType = -1,-1
            start = sum(textlenth[:ori_pos_begin])
            end = sum(textlenth[:ori_pos_end])

            if first_filter(type,sourceText,replaceText):
                continue
            else:
                if '*' in replaceText:
                    alertMessage = '含有敏感词'
                    alertMessage="含有敏感词"
                    errorType= 6
                    alertType = 4
                elif type == 'replace':
                    alertMessage = f"建议用“{replaceText}”替换“{sourceText}”"
                    alertType = 4
                    errorType = 1
                elif type == 'delete':
                    alertMessage = f"建议删除“{sourceText}”"
                    alertType = 3
                    errorType = 5
                elif type == 'insert':
                    alertMessage = f"建议添加“{replaceText}”"
                    #判断添加位置，抽取添加字符的位置，和原文进行比较
                    if ori_pos_begin == 0:
                        sourceText = source[0]
                        replacelist += sourceText
                        start = 0
                        end = textlenth[0]
                        alertType = 1
                    elif ori_pos_end == len(textlenth):
                        sourceText = source[-1]
                        start = sum(textlenth[:-1])
                        end = sum(textlenth)
                        alertType = 2
                    else:
                        sourceText = source[ori_pos_begin-1]
                        start = sum(textlenth[:ori_pos_begin-1])
                        end = sum(textlenth[:ori_pos_begin])
                        alertType = 2
                    errorType = 5
                alert_item = creat_alert_item(alertMessage, alertType, errorType, replaceText, sourceText, start, end)
                model_json.append(alert_item)
    return model_json

# 基于ltp分词对原始json经行一个扩展
def supple_error(errors,sentence):                                            
    # 用于扩展错误的信息
    # 首先是切分句子
    sentence = sentence.encode('utf-8', errors='ignore').decode('utf-8')
    textlist = ltp_filter(sentence)

    # 然后是根据切分的句字和error信息来扩展error
    textlenth = [len(text) for text in textlist]
    # if sum(textlenth) != len(sentence):
    #     logger.warning('存在问题')
    #     logger.warning(sentence)
        
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
if need_model:
    model = T5Corrector(model_dir)
ltp = LTP("LTP/base1")
if torch.cuda.is_available():
    ltp.to("cuda")



# 封装预测长文章的函数
def inferLarge(alltext):
    st = time.time()
    alerts = []
    items = []
    errorcode = 0
    errormessage = ''
    cut_text,cut_mark = long2short(alltext)
    spredTime = time.time()
    if len(cut_text):
        if len(cut_text) < 256:
            short_res = get_t5Pred(model.batch_t5_correct,cut_text)
        else:
            short_res = []
            max_l = 200
            num = math.ceil(len(cut_text)/max_l)
            for i in range(num):
                if i+1 == num:
                    temp = cut_text[i*max_l:]
                else:
                    temp = cut_text[i*max_l:(i+1)*max_l]
                tempp = get_t5Pred(model.batch_t5_correct,temp)
                short_res.extend(tempp)
    else:
        short_res = cut_text
    predTime = time.time()
    predlist = short_res
    short_pred = short2long(predlist,cut_mark,cut_text)
    logger.info(f'origin:{alltext}')
    logger.info(f'short_pred:{short_pred}')

    res = short_pred #送入后处理的预测结果
    # wrongSensitive = filter_SensitiveWord(res)
    # res = exe_filter(wrongSensitive,res)
    # logger.info(f'result:{res}')
    item = {
        'originText': alltext,
        'output': res
    }
    items.append(item) 
    try:
        alerts.append(create_json_ltp(item))
    except Exception as e:
        logger.warning('create json error')
        errorcode = 2
        errormessage = f'create json error:{e}'
        
    exeTime = time.time() - st
    predTime = predTime - spredTime
    text_lenth = len(alltext)
    preds = int(text_lenth/predTime)
    exes = int(text_lenth/exeTime)

    logger.info(f'本次，处理时间：{exeTime}s，预测时间：{predTime}s，文本长度：{text_lenth}，预测速度：{preds}字/s，处理速度：{exes}字/s。')
    speed = [exeTime,predTime,text_lenth]
    
    model_json={
        'alerts':alerts,
        'data':items,
        'errCode':errorcode,
        'errMsg':errormessage
    }

    return model_json,speed

#保存json
def saveJson(data,filePath):
    json_dicts = json.dumps(data,indent=4,ensure_ascii=False)
    with open(filePath,'w+',encoding='utf8') as fp:
        fp.write(json_dicts)

# 封装transfer功能
def post_transfer(item,filt):
    result = item['output']
    sentence = result['data']
    oritext = sentence[0]['originText']
    # ltp 切出一个结果
    alerts = second_filter(result['alerts'][0],oritext,ltp,filt)
    error = {
        'alerts':[
            alerts
        ],
        'data':[],
        "errCode": 0,
        "errMsg": ""
    }

    sen = {
        "sentences": [item['site_content']]
    }
    result = post_process_sequence(sen,error)
    try:
        
        
        alerts = result['alerts']
        output = deal_with_alert(alerts)
        response = {'detected_errors': output}
        
        return response
    except:
        return 'error'


# 用于对文本进行分词，可以返回list和str两种
def ltp_filter(sentence,need_list=True):
    sentence = sentence.encode('utf-8-sig', errors='ignore').decode('utf-8-sig')
    cut_text,cut_mark = long2short(sentence)
    def solve_ltp_error(s):
        s = unicodedata.normalize('NFKC',s)
        s = re.sub(r'\s+','',s)
        s = re.sub(r'\ufeff','',s)
        if len(s) == 0:
            return '0'
        else:
            return s

    cut_text = [solve_ltp_error(s) for s in cut_text]

    texts = []
    i = 0
    l = 300
    while i < math.ceil(len(cut_text)/l):
        if (i+1)*l >= len(cut_text):
            s = cut_text[i*l:]
        else:
            s = cut_text[i*l:(i+1)*l]
        i += 1

        result = ltp.pipeline(s,tasks=['cws'])
        texts.extend(result.cws)

    if need_list:
        textlist = []
        for m in cut_mark:
            if isinstance(m,int):
                if len(texts[m]) == 1 and texts[m][0] == '0':
                    continue
                else:
                    textlist.extend(texts[m])  
            else:
                textlist.append(m)
        return textlist
    else:
        sen = ''
        for m in cut_mark:
            if isinstance(m,int):
                if len(texts[m]) == 1 and texts[m][0] == '0':
                    continue
                else:
                    for t in texts[m]:
                        sen += t
            else:
                sen += m
        return sen

# 纠错部分的最外层代码
def run_correct_model(file_path,res_path):
    
    files = os.listdir(file_path)
    speedlist = []
    for file in files:
        path = os.path.join(file_path,file)
        print(path)
        output = []
        fp = pd.read_excel(path)
        for item in fp.iloc():
            id = str(item['id'])
            site_url = item['site_url']
            site_title = item['site_title']
            site_content = str(item['site_content'])
            site_content = ltp_filter(site_content,need_list=False)
            logger.info(f'id:{id}')
            if id == '22237':
                print('ok')
            model_json,speed = inferLarge(site_content)
            speedlist.append(speed)
            res = {
                'id':id,
                'site_url':site_url,
                'site_title':site_title,
                'site_content':site_content,
                'output':model_json
            }
            output.append(res)
        save_name = file.split('.')[0]
        save_name += '.json'
        save_path = os.path.join(res_path,save_name)
        saveJson(output,save_path)

    allExetime = 0
    allPredtime = 0
    allTextlenth = 0
    for item in speedlist:
        allExetime += item[0]
        allPredtime += item[1]
        allTextlenth += item[2]
    
    exes = allTextlenth/allExetime
    preds = allTextlenth/allPredtime
    logger.info(f'总体处理时间：{allExetime}s，预测时间：{allPredtime}s，文本长度：{allTextlenth}，预测速度：{preds}字/s，处理速度：{exes}字/s。')


# 后处理部分的最外层代码
def run_transfer_part(file_path,res_path,debug=False,debug_id=0):
    errorfile = os.path.join(res_path,'error.json')
    files = os.listdir(file_path)
    error_list = {}
    secondFilter = dictfiliter()
    for file in files:
        path = os.path.join(file_path,file)
        print(path)
        output = []
        errors = []
        with open(path,'r',encoding='utf8') as fp:
            alljson = json.load(fp)
        num = 0
        for item in alljson:
            
            id = item['id']
            site_url = item['site_url']
            site_title = item['site_title']
            site_content = item['site_content']
            print(f'{num}:{id}')
            if debug:
                if id == str(debug_id):
                    print('ok')
            num += 1
            res = post_transfer(item,secondFilter) # 进行后处理
            if res != 'error':
                temp = {
                    'id':id,
                    'site_url':site_url,
                    'site_title':site_title,
                    'site_content':site_content,
                    'detected_errors':res
                }
                output.append(temp)
            else:
                print(f'error id:{id}')
                errors.append(item)
            
        error_list[f'{file}'] = errors    
        save_name = file.split('.')[0]
        save_name += '.json'
        save_path = os.path.join(res_path,save_name)
        saveJson(output,save_path)
    saveJson(error_list,errorfile)

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

if __name__ == '__main__':
   
    
    # 数据文件名
    file_path = '区县数据'
    # ---------------创建相应文件夹-----------------
    zero_step_path = 'large_infer_outputs/originalLargeData'
    first_step_path = 'large_infer_outputs/predictLargeData'
    res_path = 'large_infer_outputs/outputLargeData'
    cut_path = 'large_infer_outputs/smalloutLargeData'
    analyse_path = 'large_infer_outputs/analyseLargeData'

    zero = os.path.join(zero_step_path,file_path)
    first = make_dir(os.path.join(first_step_path,file_path))
    res = make_dir(os.path.join(res_path,file_path))
    cut = make_dir(os.path.join(cut_path,file_path))
    ana = make_dir(os.path.join(analyse_path,file_path))

    #--------------------运行--------------------
    # 模型预测
    run_correct_model(zero,first)
    logger.info('model predict part has finished.')
    
    # 后处理
    run_transfer_part(first,res)
    logger.info('after process part has finished.')

    # 把整篇文章切成短句字，看纠错
    run_content(res,cut)
    logger.info('cut part has finished.')

    # 分析纠错情况
    run_analyse_divide(cut,ana)
    run_list_content(cut,ana)
    run_frequncefn(cut,ana)
    logger.info('analyse part has finished.')

    # 删除短句子错误，因为文件太多，使得git不好管理
    import shutil
    shutil.rmtree(cut)
    
    
    # item = {
    #     "originText":"环保总局、烟草局、民航总局等部门本级存在的",
    #     "output":"报告环保总局、烟草专卖局、中国民用航空局等部门本级存在的问题。"
    # }
    # text = '\n\t\t\t\t\t\t\t\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0 2018年第18号\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0 \xa0（二0一八年十一月公告）\n\xa0\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0 \xa0根据《中华人民共和国审计法》第二十三条规定、《关于东门街片区棚户区改造项目征收补偿资金审计的请示》（麒东门街指挥部请〔2018〕1号），麒麟区审计局派出审计组，自2018年8月23日至2018年9月7日对麒麟区东门街片区棚户区改造项目征收补偿资金进行了审计，主要审计东门街片区棚户区改造项目征收补偿资金的真实性、合法性，并对重要事项进行必要的延伸和追溯。曲靖市麒麟区东门街片区棚户区改造工作指挥部对其提供的结算资料以及其他相关资料的真实性和完整性负责。麒麟区审计局的责任是依法独立实施审计并出具审计报告。\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0 \xa0一、基本情况\n\xa0\xa0\xa0\xa0\xa0\xa0 （一）立项及投资批复情况\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0 曲靖市麒麟区东门街片区棚户区改造项目，是根据《曲靖市麒麟区人民政府关于曲靖市麒麟区东门街片区棚户区改造房屋征收的公告》（麒区政公告〔2015〕4号） 、《曲靖市麒麟区人民政府关于曲靖市麒麟区东门街片区棚户区改造房屋征收的决定》、《曲靖市麒麟区东门街片区棚户区改造房屋征收补偿方案》实施的。征收主体：曲靖市麒麟区人民政府；征收部门：曲靖市麒麟区南宁街道办事处；征收实施单位：麒麟区南宁街道办事处潇湘社区居委会。征收范围和规模：东至紫云路、西至麒麟区南路、南至潇湘路、北至文昌街延长线范围内的房屋，以及该范围内的所有建（构）筑物、附着物等。\n\xa0\xa0\xa0\xa0\xa0\xa0 （二）项目建设实施情况\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0 该项目2015年3月9日成立曲靖市麒麟区东门街片区棚户区改造工作指挥部。2016年1月开始实施房屋征收工作，由南宁街道办事处、南宁街道潇湘社区居委会负责具体实施。项目融资贷款、安置房建设及配套基础设施建设由西南交通建设集团股份有限公司负责实施。麒麟区城市建设投资有限公司负责项目实施中资金流转、缺口资金筹措事项。\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0 二、审计情况\n\xa0\xa0\xa0\xa0\xa0\xa0 根据曲靖市麒麟区东门街片区棚户区改造工作指挥部及曲靖市城市建设投资有限公司、麒麟区南宁街道办事处提供的资料显示：\n\xa0\xa0\xa0\xa0\xa0\xa0 截止到2018年8月31日，该项目共筹集资金14.83亿元。其中：西南交通建设集团股份有限公司向国家开发银行股份有限公司贷款14.89亿元，实际到位资金10.05亿元；筹集资本金14833.88万元（其中：央补省补4227.88万元；农发行专项建设基金8606万元；麒麟区城投公司自筹资金2000万元）；麒麟区城市建设投资开发有限公司自筹资金3.3亿元。\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0 截止到2018年8月31日，曲靖市麒麟区东门街片区棚户区改造项目已签订私户房屋征收补偿协议2311户，签约率达到98.8%，拆除房屋约21万平方米。\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0 三、审计结果\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0 截止2018年8月31日，曲靖市麒麟区东门街片区棚户区改造项目共支出征收补偿资金1,178,701,794.00元。\n\ufeff\xa0\xa0\xa0\xa0\xa0\xa0\xa0 四、审计评价意见\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0 该项目拆迁补偿资金是按有关房屋拆迁补偿估价报告评估价及房屋拆迁补偿安置协议进行补偿，补偿标准是按《曲靖市麒麟区东门街片区棚户区改造房屋征收补偿方案》规定所签定的协议进行补偿。\n\xa0\xa0\xa0\xa0\xa0\xa0\xa0 五、审计建议\n\xa0\xa0\xa0\xa0\xa0 （一）严格执行项目征收补偿相关程序，按照征收补偿方案进行补偿。\n\xa0\xa0\xa0\xa0\xa0 （二）注意防范棚户区改造项目贷款带来的政府性债务风险。\n\xa0\xa0\xa0\xa0\xa0 （三）做好项目资料的归档工作。\n\xa0\xa0\xa0\xa0\xa0\xa0 对此次审计情况，曲靖市麒麟区东门街片区棚户区改造工作指挥部高度重视，认真组织落实，审计建议已采纳。\n\t\t\t\t\n\t\t\t\t                \n\t\t\t\t\t\t'
    # res = ltp_filter(text,False)
    # print(res)