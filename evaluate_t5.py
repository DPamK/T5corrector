import os
import json
from loguru import logger
import re
import time
import math
from utils.fixModeloutput import E_trans_to_C
from transformers import AutoTokenizer, T5ForConditionalGeneration
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_t5Pred(correct_fn,sources):
    res = correct_fn(sources)
    return res

def delete_blank(text):
    s = re.sub('[ \001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\u0000\u3000\xa0\xa0]', '', text)
    return s

def get_errors(corrected_text):
    return E_trans_to_C(corrected_text)

class T5Corrector(object):
    def __init__(self,model_dir='output/t5_newdata'):
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
            corrected_text = get_errors(corrected_text)
            text_new += corrected_text
            result.append(text_new)
        return result

#读取json
def readJson(filePath):
    with open(filePath, 'r', encoding='utf8') as f:
        jsondata = json.load(f)
    return jsondata

#保存json
def saveJson(data,filePath):
    json_dicts = json.dumps(data,indent=4,ensure_ascii=False)
    with open(filePath,'w+',encoding='utf8') as fp:
        fp.write(json_dicts)

# 创建文件夹
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def eval_model_prf(data,srcname,trgname,predname,verbose=True):
    """
    评估结果，设定需要纠错为正样本，无需纠错为负样本
    params:
        data:验证集预测后的结果字典
        srcname:原始文本的key
        trgname:目标文本的key
        predname:预测文本的key
        verbose:

    Returns:
        Acc, Recall, F1
        检错正确率
        纠错正确率
    """
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    res = []
    srcs = []
    tgts = []
        
        
    for item in data:
        srcs.append(delete_blank(item[srcname]))
        tgts.append(delete_blank(item[trgname]))
        res.append(delete_blank(item[predname]))

    collection_wrong = 0
    collection_right = 0

    right_de_right = 0
    right_de_wrong = 0
    wrong_de_right = 0
    wrong_de_wrong = 0
    de_co_res = []
    # f1 = open(result_path, 'w', encoding='utf-8')
    f1 = []
    for tgt_pred, src, tgt in zip(res, srcs, tgts):
        if verbose:
            # 正确样本
            if src == tgt:
                # 预测也为正
                if tgt == tgt_pred:
                    TN += 1
                    right_de_right += 1
                    # resul_str = 'input:' + src + '\t' + 'target:' + tgt + '\t' + 'pred:' + tgt_pred + '\t' + 'right' + '\n'
                    resul_str = 'right'
                    f1.append(resul_str)
                    de_co_res.append([False,False])
                    
                # 预测为错
                else:
                    FP += 1
                    right_de_wrong += 1
                    # resul_str = 'input:' + src + '\t' + 'target:' + tgt + '\t' + 'pred:' + tgt_pred + '\t' + 'wrong' + '\n'
                    resul_str = 'wrong'
                    f1.append(resul_str)
                    de_co_res.append([False,False])
                    
            # 错误样本
            else:
                # 预测也为错误,且与tgt一致
                if tgt == tgt_pred:
                    TP += 1
                    wrong_de_wrong += 1
                    collection_right += 1
                    # resul_str = 'input:' + src + '\t' + 'target:' + tgt + '\t' + 'pred:' + tgt_pred + '\t' + 'right' + '\n'
                    resul_str = 'right'
                    f1.append(resul_str)
                    s = [True,True]
                    de_co_res.append(s)
                # 预测与tgt不一致
                else:
                    FN += 1
                    # resul_str = 'input:' + src + '\t' + 'target:' + tgt + '\t' + 'pred:' + tgt_pred + '\t' + 'wrong' + '\n'
                    resul_str = 'wrong'
                    f1.append(resul_str)
                    if tgt_pred == src:
                        # 没有检测出错误
                        wrong_de_right += 1
                        de_co_res.append([False,False])  
                    else:
                        # 检查出错误，但没有改对
                        wrong_de_wrong += 1
                        collection_wrong += 1
                        s = [True,False]
                        de_co_res.append(s)
            total_num += 1
    res = f1

    '''
    直观判断：检测错误/检测正确，纠错错误/纠错正确
    ''' 
    wrong_sample = wrong_de_right + wrong_de_wrong
    right_sample = right_de_right + right_de_wrong
    wrong_de_acc = wrong_de_wrong / wrong_sample if wrong_sample > 0 else 0.0
    right_de_acc = right_de_right / right_sample if right_sample > 0 else 0.0
    detection_right = right_de_right + wrong_de_wrong
    detection_wrong = right_de_wrong + wrong_de_right

    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    colusion = f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, \n \
                TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN} \n \
                正确样本共 {right_sample} 例，其中检测正确：{right_de_right}，检测错误：{right_de_wrong}，准确率：{right_de_acc} \n \
                错误样本共 {wrong_sample} 例，其中检测正确：{wrong_de_wrong}，检测错误：{wrong_de_right}，准确率：{wrong_de_acc} \n \
                纠错正确：{collection_right}，纠错错误：{collection_wrong}。 \n'
    
    print(
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
        f'检测正确：{detection_right}，检测错误：{detection_wrong}，纠错正确：{collection_right}，纠错错误：{collection_wrong}')
    # return acc, precision, recall, f1
    return colusion,res,de_co_res

def eval_SQyuliao(model_dir,evaloutPATH):
    '''
    功能：去测试数起给定的语料的infer结果，只针对长文本，短文本慎用
    参数: 
        model_dir:模型位置
        evaloutPATH:评价结果保存位置
    返回：生成两个文件，一个是所有预测的结果json，一个是预测错误的情况统计tsv
    '''
    data_path = 'test_data/shenji_yuliao.json'
    data_num = 100
    make_dir(evaloutPATH)
    model_name = model_dir.split('/')[-1]
    res_path = os.path.join(evaloutPATH,model_name+'_yuliao_infer.json')
    tsv_path = os.path.join(evaloutPATH,model_name+'_yuliao_inferwrong.tsv')
    jsondata = readJson(data_path)
    tests_num = data_num
    res = []
    lenth = len(jsondata)
    i = 0
    model = T5Corrector(model_dir)
    batch_num = math.ceil(lenth/tests_num)
    logger.info(f'len={lenth}, batch_num={batch_num}')
    for j in range(batch_num):
        if (j+1)*tests_num >= lenth:
            tests = jsondata[j*tests_num:]
        else:
            tests = jsondata[j*tests_num:(j+1)*tests_num]
        srcs = []
        for item in tests:
            srcs.append(item['wrongText'])
        t5_preds = get_t5Pred(model.batch_t5_correct,srcs)
        for item,pred in zip(tests,t5_preds):
            item['t5_revised'] = pred
            res.append(item)
    print(len(res))
    saveJson(res,res_path)
    prf,Res,deco = eval_model_prf(res,'wrongText','rightTextR','t5_revised')
    with open(tsv_path,'w+',encoding='utf8') as f:
        f.write(prf)
        for data,r in zip(res,deco):
            if r[0] and not r[1]:
                temp = 'input:\t' + data['wrongText'] + '\n' + 'target:\t' + data['rightTextR'] + '\n' + 't5_pred:' + data['t5_revised'] + '\n' 
                f.write(temp)

def eval_vaild(model_dir,evaloutPATH,data_path,data_num=200):
    '''
    功能：去测试 验证集vaild的infer结果
    参数: 
        model_dir:模型位置
        evaloutPATH:评价结果保存位置
        data_path:验证集文件位置
        data_num:预测时batch大小
    返回：生成两个文件，一个是所有预测的结果json，一个是预测错误的情况统计tsv
    '''
    dataname = data_path.split('/')[-1].split('.')[0]
    make_dir(evaloutPATH)
    model_name = model_dir.split('/')[-1]
    res_path = os.path.join(evaloutPATH,model_name+'_yuliao_infer.json')
    tsv_path = os.path.join(evaloutPATH,model_name+'_yuliao_inferwrong.tsv')
    jsondata = readJson(data_path)
    tests_num = data_num
    res = []
    lenth = len(jsondata)
    i = 0
    model = T5Corrector(model_dir)
    batch_num = math.ceil(lenth/tests_num)
    logger.info(f'len={lenth}, batch_num={batch_num}')
    for j in range(batch_num):
        if (j+1)*tests_num >= lenth:
            tests = jsondata[j*tests_num:]
        else:
            tests = jsondata[j*tests_num:(j+1)*tests_num]
        srcs = []
        for item in tests:
            srcs.append(item['wrongText'])
        t5_preds = get_t5Pred(model.batch_t5_correct,srcs)
        for item,pred in zip(tests,t5_preds):
            item['t5_revised'] = pred
            res.append(item)
    print(len(res))
    saveJson(res,res_path)
    prf,Res,deco = eval_model_prf(res,'wrongText','rightTextR','t5_revised')
    with open(tsv_path,'w+',encoding='utf8') as f:
        f.write(prf)
        for data,r in zip(res,deco):
            if r[0] and not r[1]:
                temp = 'input:\t' + data['wrongText'] + '\n' + 'target:\t' + data['rightTextR'] + '\n' + 't5_pred:' + data['t5_revised'] + '\n' 
                f.write(temp)




if __name__=="__main__":
    model_dir = 'output/t5_newdata_1009'
    savepath = 'evalouts/eval_1014'

    eval_SQyuliao(model_dir,savepath)








