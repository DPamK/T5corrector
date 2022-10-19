import json
import requests
import re
import pandas as pd
import cn2an
import sqlite3
# from LAC import LAC

# lac = LAC(mode="lac")
# from paddlenlp import Taskflow
# lac = Taskflow("ner")
from ltp import LTP

ltp = LTP("LTP/base1")

dict_path = './dict/'
# leader_info = pd.read_excel(dict_path + 'leader.xls')

# with open(dict_path + 'black_word.txt', 'r', encoding='utf8') as black_word_file:
#     black_words = [str(black_word).strip() for black_word in black_word_file.readlines()]
#
# with open(dict_path + 'black_pair.txt', 'r', encoding='utf8') as black_pair_file:
#     black_pairs = [str(black_pair).strip() for black_pair in black_pair_file.readlines()]
#
# with open(dict_path + 'slogan.txt', 'r', encoding='utf8') as slogan_file:
#     slogans = [str(slogan).strip() for slogan in slogan_file.readlines()]

with open(dict_path + 'location.json', 'r', encoding='utf8') as location_file:
    location_structure = json.load(location_file)


# with open(dict_path + 'idiom.txt', 'r', encoding='utf8') as idiom_file:
#     idioms = [str(idiom).strip() for idiom in idiom_file.readlines()]


def post_disable_repeat(model_json):
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if alert_error['alertType'] != 16:
                new_alert_error = dict(alert_error)
                new_alert_error["alertType"] = 16

                for alert_error_2 in sentence_errors:
                    if alert_error_2['start'] == new_alert_error['start'] \
                            and alert_error_2['end'] == new_alert_error['end'] \
                            and alert_error_2['alertType'] == new_alert_error['alertType']:
                        sentence_errors.remove(alert_error_2)
                        break

    return model_json


def post_disable_by_black_words(model_json, black_words):
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if alert_error['sourceText'] in black_words:
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_by_black_pairs(model_json, black_pairs):
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            pair = str(alert_error['sourceText']).strip() + ' ' + str(alert_error['replaceText'])
            if pair in black_pairs:
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_by_sentence_slogan(input_sentence, model_json, slogans):
    for sentence, sentence_errors in zip(input_sentence['sentences'], model_json['alerts']):
        slogan_ranges = []
        slogan_existing = False
        for slogan in slogans:
            if slogan in sentence:
                slogan_existing = True
                slogan_start = sentence.index(slogan)
                slogan_end = slogan_start + len(slogan) - 1
                slogan_ranges.append((slogan_start, slogan_end))
        if slogan_existing:
            for alert_error in list(sentence_errors):
                for slogan_range in slogan_ranges:
                    slogan_start, slogan_end = slogan_range
                    if alert_error['start'] >= slogan_start and alert_error['end'] <= slogan_end:
                        sentence_errors.remove(alert_error)
                        break
    return model_json


def post_enable_location(input_sentence, model_json, location_structure):
    """
        根据原文查找202错误：对行政区划的层级进行判断处理，
        分析主要的问题是：检测出模型未检出的202错误，并提示。
        :param model_json:
        :return:
        """
    p = re.compile(r"[\u4e00-\u9fa5]{1,4}[省市县区]")
    p0 = re.compile(r"[\u4e00-\u9fa5]+省((?!市).)*区")
    location_hashmap = create_gov_level_hash(location_structure)
    for sentence, sentence_error in zip(input_sentence['sentences'], model_json['alerts']):
        errorList = re.findall(p, sentence)
        errorList = delete_conjunctions(errorList)
        errorList = [clean_location_name(name,location_hashmap) for name in errorList]
        for text in errorList:
            if 'error' in text:
                # text = text[5:]
                # if not is_same_location_error_exist(text, sentence_error):
                #     start = sentence.find(text)
                #     item = createAdvise(text, start, '', '地址错误')
                #     sentence_error.append(item)
                continue
            if re.match(p0, text) :
                if is_process_needed(text, location_structure):
                    if not is_same_location_error_exist(text, sentence_error):
                        start = sentence.find(text)
                        item = createAlert(text, start, location_structure)
                        sentence_error.append(item)
                else:
                    if not is_same_location_error_exist(text, sentence_error):
                        start = sentence.find(text)
                        item = createAdvise(text, start, '', '地址错误')
                        sentence_error.append(item)

            if re.match(r"([\u4e00-\u9fa5]+省)?[\u4e00-\u9fa5]+市([\u4e00-\u9fa5]+[区县])?",text):
                exist_wrong, revise = wrong_gov_level(text,location_hashmap)
                if exist_wrong:
                    if not is_same_location_error_exist(text, sentence_error):
                        start = sentence.find(text)
                        item = createAdvise(text, start, revise,'归属地错误')
                        sentence_error.append(item)

    return model_json

def delete_conjunctions(errorlist):
    res = []
    for error in errorlist:
        if exist_conjunctions(error):
            temp = re.split('和|及|与|兼|跟|在|或',error)
            for m in temp:
                res.append(m)
        else:
            res.append(error)
    return res

def exist_conjunctions(text):
    words = ['和','及','与','兼','跟','在','或']
    citys = ['呼和浩特市','和县','和区','和田']
    for word in words:
        if word in text :
            for c in citys:
                if c in text:
                    return False
            return True
    return False

def clean_location_name(text,hashmap):
    p = re.compile(r"(?<=[省市县区])")
    place = re.split(p, text)
    place.pop()
    if '城区' in place:
        place.remove('城区')
    lenth = len(place)
    res = ''
    for i in range(lenth):
        newp = catch_location_name(place[i], hashmap)
        if newp == 'error':
            return 'error'+text
        else:
            res += newp
    return res


def createAdvise(text, start, revise, message):
    res = {
        'advancedTip': True,
        'message': message,
        'alertType': 4,
        'end': start + len(text) - 1,
        'errorType': 202,
        'replaceText': revise,
        'sourceText': text,
        'start': start
    }
    return res


def catch_location_name(text, location_hashmap):
    '''
    因为切词的不严谨会导致多切一些内容，所以需要修正一下行政名
    :param text:
    :param location_hashmap:
    :return: 正确的行政名
    '''
    if text[-1] == '省':
        if text in location_hashmap['p2c']:
            return text
        else:
            for province_name in location_hashmap['p2c'].keys():
                if province_name in text:
                    text = province_name
                    return text
            return 'error'
    elif text[-1] == '市':
        if text in location_hashmap['c2co']:
            return text
        else:
            for city_name in location_hashmap['c2co'].keys():
                if city_name in text:
                    text = city_name
                    return text
            return 'error'
    elif text[-1] == '区':
        if text in location_hashmap['co2c']:
            return text
        else:
            for name in location_hashmap['co2c'].keys():
                if name in text:
                    text = name
                    return text
            return 'error'
    elif text[-1] == '县':
        if text in location_hashmap['co2c']:
            return text
        else:
            for name in location_hashmap['co2c'].keys():
                if name in text:
                    text = name
                    return text
            return 'error'
    else:
        return 'error'


def wrong_gov_level(text, location_hashmap):
    '''
    返回是否出现归属地错误，且返回修改内容
    :param text:
    :param location_hashmap:
    :return: bool，text
    '''
    p = re.compile(r"(?<=[省市区县])")
    place = re.split(p, text)
    place.pop()
    lenth = len(place)
    for i in range(lenth):
        newp = catch_location_name(place[i], location_hashmap)
        if newp == 'error':
            return False, ''
        else:
            place[i] = newp
    if lenth == 2:
        if place[0][-1] == '省':
            if place[1] in location_hashmap['p2c'][place[0]]:
                return False, ''
            else:
                if place[1] not in location_hashmap['c2p'] and place[1] in location_hashmap['c2co']:
                    return True, place[1]
                else:
                    maybep = location_hashmap['c2p'][place[1]][0]
                    return True, maybep + place[1]
        elif place[0][-1] == '市':
            if place[1] in location_hashmap['c2co'][place[0]]:
                return False, ''
            else:
                maybec = location_hashmap['co2c'][place[1]][0]
                return True, maybec + place[1]
        else:
            return False, ''

    elif lenth == 3:
        if place[0] in location_hashmap['p2c']:
            if place[1] in location_hashmap['p2c'][place[0]]:
                if place[2] in location_hashmap['c2co'][place[1]]:
                    return False, ''
                else:
                    if place[2] in location_hashmap['co2c']:
                        maybe_city = location_hashmap['co2c'][place[2]]
                        for mc in maybe_city:
                            if mc in location_hashmap['p2c'][place[0]]:
                                return True, place[0] + mc + place[2]
                    else:
                        return True, place[0] + place[1]
            else:
                if place[2] in location_hashmap['co2c']:
                    maybe_city = location_hashmap['co2c'][place[2]]
                    for mc in maybe_city:
                        if mc not in location_hashmap['c2p'] and mc in location_hashmap['c2co']:
                            return True, mc + place[2]
                        elif mc in location_hashmap['p2c'][place[0]]:
                            return True, place[0] + mc + place[2]

                    cy = maybe_city[0]
                    pv = location_hashmap['c2p'][cy][0]
                    return True, pv + cy + place[2]
                else:
                    return False, ''
        else:
            return False, ''
    else:
        return False, ''


def create_gov_level_hash(location_structure):
    '''
    创建hash表，p2c对应省到市,c2p对应市到省,c2co对应市到区，co2c对应区到市
    :param location_structure:
    :return: hashmap{'p2c','c2p','c2co','co2c'}
    '''
    province_city = {}
    city_province = {}
    city_county = {}
    county_city = {}
    for province in location_structure:
        for city in province['city']:
            if city['name'] != '市辖区' and city['name'] != '县':
                for county in city['county']:

                    if city['name'] not in city_county:
                        city_county[city['name']] = [county]
                    else:
                        city_county[city['name']].append(county)

                    if county not in county_city:
                        county_city[county] = [city['name']]
                    else:
                        county_city[county].append(city['name'])
            else:
                for county in city['county']:
                    if province["province"] not in city_county:
                        city_county[province["province"]] = [county]
                    else:
                        city_county[province["province"]].append(county)

                    if county not in county_city:
                        county_city[county] = [province["province"]]
                    else:
                        county_city[county].append(province["province"])

            if city['name'] != '市辖区' and city['name'] != '县':
                if province["province"] not in province_city:
                    province_city[province["province"]] = [city['name']]
                else:
                    province_city[province["province"]].append(city['name'])

                if city['name'] not in city_province:
                    city_province[city['name']] = [province["province"]]
                else:
                    city_province[city['name']].append(province["province"])
    res = {
        'p2c': province_city,
        'c2p': city_province,
        'c2co': city_county,
        'co2c': county_city
    }
    return res


def is_process_needed(text, location_structure):  # 新增
    p = re.compile(r"(?<=[省区])")
    place = re.split(p, text)  # 将词条切分为：XX省，XX市，XX县
    place.pop()
    for province in location_structure:
        if province['province'] == place[0]:
            for city in province['city']:
                if place[1] in city["county"]:
                    return True
            return False


def is_same_location_error_exist(text, sentence_error):
    res = False
    for item in sentence_error:
        if text in item['sourceText']:
            return True
    return res


def createAlert(text, start, location_structure):
    res = {
        'advancedTip': True,
        'message': '行政区划错误',
        'alertType': 4,
        'end': start + len(text) - 1,
        'errorType': 202,
        'replaceText': revise_202(text, location_structure),
        'sourceText': text,
        'start': start
    }
    return res


def revise_202(text, location_structure):  # 针对XX省XX区的错误

    p = re.compile(r"(?<=[省市县区镇村])")
    place = re.split(p, text)  # 将词条切分为：XX省，XX市，XX县
    place.pop()
    for province in location_structure:
        if province['province'] == place[0]:
            for city in province['city']:
                if place[1] in city["county"]:
                    return place[0] + city['name'] + place[1]
    return text


def post_disable_local(model_json):
    """
    针对C1错误：对行政区划的层级进行判断处理，
    分析主要的问题是：模型中会将县级市和县两个行政等级补全，这个功能不需要。
    解决方法：使用逻辑判断是否需要删除
    :param model_json:
    :return:
    """

    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if alert_error['errorType'] == 202:
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_context(model_json):
    '''
    C6 无意义修改词，未按语境理解词义
    “审计”->"升级，省级..."
    '''

    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            error_word = alert_error['replaceText']
            if alert_error['sourceText'][:2] == "审计" and error_word[:2] == "审计":
                print(alert_error['sourceText'][:2])
                continue
            elif error_word[:2] == "审计":
                continue
            elif alert_error['sourceText'][:2] == "审计" and error_word[:2] != "审计":
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_gongwenguifan(model_json):
    '''
    可能是词库硬匹配造成的
    提供专有名称给算法（目前不多）
    中国特色社会主义审计制度、中国特色社会主义审计事业、中国特色社会主义审计道路、合同法、合同工、
    贯彻落实党中央各项决策部署、贯彻落实审计署决策部署
    '''
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            error_word = alert_error['replaceText']
            error_type = alert_error['alertMessage']
            if error_type[0:8] == '建议使用公文规范':
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_space_after_puncs(model_json):
    '''
    "中文逗号（冒号、引号）,
    建议修改成中文逗号（冒号、引号）,
    建议修改和原始标点符号是相同的
    '''
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if alert_error['errorType'] == 2 and alert_error['alertMessage'] == '中文符号后不空格':
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_space(model_json):
    '''
    中间有空格的会识别出错误
    '''
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            error_cla = alert_error['sourceText']
            if len(error_cla) == 3:
                if error_cla[1] == ' ' or error_cla[1] == ' ':
                    sentence_errors.remove(alert_error)
    return model_json


def post_disable_number(model_json):
    """
    针对C12错误：阿拉伯数组转换为汉语
    :param model_json:
    :return:
    """
    p = re.compile(r"[0-9]+[多几]+")  # 存储对应的正则表达式
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if re.match(p, alert_error['sourceText']):
                sentence_errors.remove(alert_error)
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            source_text = alert_error['sourceText']
            replace_text = alert_error['replaceText']
            if source_text[0] == '第' and source_text[-1] == '届' and replace_text[0] == '第' and replace_text[
                -1] == '届':
                source_number = int(source_text[1:-1])
                replace_number = cn2an.cn2an(replace_text[1:-1], "normal")
                if source_number == replace_number:
                    sentence_errors.remove(alert_error)
    return model_json


# def post_c1(model_json):
#     """
#     针对C1错误：对一些专有地名进行忽略处理
#     :param model_json:
#     :return:
#     """
#     for sentence_errors in model_json['alerts']:
#         for alert_error in list(sentence_errors):
#             if alert_error['errorType'] == 202:
#                 sentence_errors.remove(alert_error)
#     return model_json


def post_disable_bookname(input_data, model_json):
    """
    把书名号错误，取消处理
    :param model_json:
    :return:
    """
    len_book = 50
    for sentence, sentence_errors in zip(input_data['sentences'], model_json['alerts']):
        for alert_error in list(sentence_errors):
            start = alert_error['start']
            end = alert_error['end']
            sentence_range = sentence[max(0, start - len_book):min(len(sentence), end + len_book + 1)]
            res = re.finditer('《(.*?)》', sentence_range)
            for bookidx in res:
                start_idx, end_idx = bookidx.span()
                if max(0, start - len_book) + start_idx <= start and max(0, start - len_book) + end_idx > end:
                    sentence_errors.remove(alert_error)
                    break
    return model_json


def post_disable_date(model_json):
    """
    把日期错误，取消处理
    :param model_json:
    :return:
    """
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if alert_error['errorType'] != 2:
                continue
            s_parts = alert_error['sourceText'].split('.')
            if len(s_parts) <= 2:
                is_digit = True
                for part in s_parts:
                    if not part.isdigit():
                        is_digit = False
                if is_digit:
                    sentence_errors.remove(alert_error)
    return model_json


# def post_disable_symbol(input_data, model_json):
#     """
#     把符号间隔错误，取消处理
#     :param model_json:
#     :return:
#     """
#     k = 0
#     for sentence_errors in model_json['alerts']:
#
#         for alert_error in list(sentence_errors):
#             last_inx = alert_error['start'] - 1
#             if last_inx >= 0:
#                 last_word = input_data['sentences'][k][last_inx]
#                 if last_word in '，、；。：':
#                     sentence_errors.remove(alert_error)
#         model_json['alerts'][k] = [error for error in sentence_errors if error is not None]
#         k += 1
#
#     return model_json


def post_disable_rongyu(model_json):
    """
    字词错误，c13.1 和 c7类型都能处理
    取消修改
    :param model_json:
    :return:
    """
    target_string = ['冗余', '罐']  # 冗余是指alertMessage的语义冗余提示， 罐 是针对客户提出的一个特定用例
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if 'replaceText' in alert_error.keys():
                for string in target_string:
                    if string in alert_error['alertMessage'] or string == alert_error['sourceText']:
                        sentence_errors.remove(alert_error)
                        break
            else:
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_nobody(input_sentence, model_json, leader_name_set, leader_position_set, leader_position_dict):
    """
        去掉没有职务前缀的人名错误提示
        :param model_json:input:原文
        :return:
        """

    for sentence, sentence_errors in zip(input_sentence['sentences'], model_json["alerts"]):
        names = extract_name(sentence)
        for name, start, end in names:
            left_border = start - 10 if start - 10 > 0 else 0
            right_border = end + 4
            errorType, message, hit_position, replaceText = judge_exit(name, sentence[left_border:right_border],
                                                                       leader_name_set,
                                                                       leader_position_set, leader_position_dict)

            for alert_error in list(sentence_errors):
                if name in str(alert_error['sourceText']).strip() and alert_error['errorType'] != errorType:
                    sentence_errors.remove(alert_error)
                if hit_position != '' and hit_position in alert_error['sourceText']:
                    if name in alert_error['sourceText'] and name in alert_error['replaceText']:
                        if alert_error in sentence_errors:
                            sentence_errors.remove(alert_error)

            if replaceText != "":  # 领导人名字写错的时候为空
                new_item = creat_name_item(name, start, end, errorType, message, replaceText)
                for alert_error in sentence_errors:
                    if name in alert_error['sourceText'] and "，" in alert_error["sourceText"]:  # sourceText中有人名和逗
                        alert_error["alertMessage"] = alert_error["alertMessage"].replace(alert_error['sourceText'],
                                                                                          name)
                        alert_error["sourceText"] = name
                    elif name == alert_error['sourceText']:  # sourceText和人名完全相等
                        alert_error = new_item
                sentence_errors.append(new_item)

    return model_json


def extract_name(sentence: str):
    namelist = []
    result = ltp.pipeline([sentence], tasks=['cws', 'pos'])
    seg = result.cws[0]
    pos = result.pos[0]
    current_pos = 0
    for token, label in zip(seg, pos):

        if label == 'nh':
            if bool(re.search(r'\d', token)):
                continue
            start = current_pos
            end = start + len(token) - 1
            name_tuple = (token, start, end)
            namelist.append(name_tuple)
        current_pos += len(token)
    return namelist


def creat_name_item(name, start, end, errorType, message, replaceText):
    res = {
        'advancedTip': True,
        'message': message,
        'alertType': 10,
        'end': end,
        'errorType': errorType,
        'replaceText': replaceText,
        'sourceText': name,
        'start': start
    }
    return res


def judge_exit(name_o, string_before_name, name_set, position_set, leader_position_dict):
    leader_position_list = leader_position_dict[name_o] if name_o in name_set else []
    hit_position = ''
    replaceText = ''
    for position in position_set:
        if position in string_before_name:  # 句子前面有职务
            print(position)
            hit_position = position
            for leader_position in leader_position_list:
                if leader_position in string_before_name:  # 句子前面的职务正确
                    errorType = 667
                    message = "领导职务，请谨慎查验"
                    replaceText = name_o
                    return errorType, message, hit_position, replaceText
            if len(leader_position_list) > 0:  # 职务不正确：李克强总书记
                errorType = 201
                message = "职务可能有误，建议修改为:" + '、'.join(set(leader_position_list))
                replaceText = name_o
                return errorType, message, hit_position, replaceText
            elif len(leader_position_list) == 0:  # 人名不正确：习进平总书记
                erroType = 201
                message = "领导人名可能有误"

                return erroType, message, hit_position, replaceText
    errorType = 667  # 句子没有职务：张海波
    message = "人名，请谨慎查验"
    replaceText = name_o
    return errorType, message, hit_position, replaceText


def post_disable_low(model_json):
    """
       删掉 法律名称简写不能使用书名号C12.4
       :param model_json:
       :return:
       """
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            message = alert_error['alertMessage']
            for j in range(0, len(message)):
                if message[j:] == "法律法规简称一般推荐不带书名号。":
                    sentence_errors.remove(alert_error)
    return model_json


def post_disable_reverse_words(model_json):
    """
    删掉  词序颠倒C6.6
    :param
    :return:
    """
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            replaceText = list(alert_error['replaceText'])
            sourceText = list(alert_error['sourceText'])
            replaceText.sort()
            sourceText.sort()
            if sourceText == replaceText:
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_error_quotation(input_sentence, model_json):
    """
      单双引号
      :param model_json:input:字典
      :return:
      """

    def check_quotation(sentence, start_idx):
        return len(re.findall("“", sentence[0: start_idx])) % 2 == 0

    # 遍历每个sentence 的报错列表
    for sentence, sentence_errors in zip(input_sentence["sentences"], model_json['alerts']):
        # 获取当前数据源句子
        # 遍历当前sentence的 error
        for alert_error in list(sentence_errors):
            if alert_error['alertMessage'] == "建议使用双引号" and not check_quotation(sentence, alert_error["start"]):
                sentence_errors.remove(alert_error)

    return model_json


def post_disable_remove_error_quotes(model_json):
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if alert_error['sourceText'] == '“' + alert_error['replaceText'] + '”':
                sentence_errors.remove(alert_error)
    return model_json


def correct_location_error(model_json, location_structure):
    """
            针对中国石油云南分公司的错误，
            解决方法：source是 XX+某省，replace是 某省的，比如 大美山东 -> 山东省， 去掉错误
            :param model_json:
            :return:
            """
    pr = re.compile(r'[\u4e00-\u9fa5]{1,4}省$')
    p = re.compile(r'省')
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if re.match(pr, alert_error['replaceText']) and \
                    is_province_location(alert_error['replaceText'], location_structure):
                province = re.split(p, alert_error['replaceText'])[0]
                ps = re.compile(r'[\u4e00-\u9fa5]+' + province + r'$')
                if re.match(ps, alert_error['sourceText']):
                    sentence_errors.remove(alert_error)
    return model_json


def is_province_location(text, location_structure):
    for province in location_structure:
        if province['province'] == text:
            return True
    return False


def post_disable_empty_replace_text(model_json):
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if 'replaceText' not in alert_error.keys():
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_idioms(model_json, idioms):
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if alert_error['sourceText'] in idioms and alert_error['replaceText'] in idioms:
                sentence_errors.remove(alert_error)
    return model_json


def post_enable_mix_pair_symbol_detection(input_data, model_json):
    mix_pairs = {'（': ')', '(': '）', '［': ']', '[': '］', '｛': '}', '{': '｝', '《': '>', '<': '》'}
    english_to_chinese_pair = {"(": "（", ")": "）", "[": "［", "]": "］", "{": "｛", "}": "｝", "<": "《", ">": "》"}
    mix_pairs_inverse = {v: k for k, v in mix_pairs.items()}
    quotes_list = ['"', "'", '“', '”', '‘', '’']

    for sentence, sentence_errors in zip(input_data['sentences'], model_json['alerts']):
        for alert_error in list(sentence_errors):
            if '成对' in alert_error['alertMessage']:
                if alert_error['sourceText'] in mix_pairs_inverse.keys():
                    start = alert_error['start']
                    anchor_symbol = mix_pairs_inverse[alert_error['sourceText']]
                    mix_pair_flag = False
                    for i in range(0, start):
                        current_index = start - i
                        if sentence[current_index] == anchor_symbol:
                            alert_error['start'] = current_index
                            alert_error['end'] = current_index
                            alert_error['sourceText'] = anchor_symbol
                            alert_error['replaceText'] = english_to_chinese_pair[anchor_symbol]
                            alert_error['alertType'] = 4
                            mix_pair_flag = True
                            break
                    if not mix_pair_flag:
                        alert_error['message'] = '缺乏成对的标点符号'
                        alert_error['alertType'] = 10
                elif alert_error['sourceText'] in quotes_list:
                    alert_error['message'] = '缺乏成对的标点符号'
                    alert_error['alertType'] = 10
                else:
                    sentence_errors.remove(alert_error)

    return model_json


def fetch_data_by_type(db_connect, data_type):
    cursor = db_connect.cursor()
    cursor.execute('select * from ' + data_type)
    rows = cursor.fetchall()

    return rows


def get_db_dict_assets(db_connect):
    black_pairs = fetch_data_by_type(db_connect, 'black_pairs')#匹配两个
    black_pairs = set([pair[1] + ' ' + pair[2] for pair in black_pairs])

    black_words = fetch_data_by_type(db_connect, 'black_words')#匹配source
    black_words = set([word[1] for word in black_words])

    idioms = fetch_data_by_type(db_connect, 'idioms')
    idioms = set([idiom[1] for idiom in idioms])

    leader_infos = fetch_data_by_type(db_connect, 'leaders')
    leader_name_set = set([leader[1] for leader in leader_infos])
    leader_position_set = set([leader[2] for leader in leader_infos])

    slogans = fetch_data_by_type(db_connect, 'slogans')
    slogans = set([slogan[1] for slogan in slogans])

    return black_pairs, black_words, idioms, leader_name_set, leader_position_set, slogans


def build_leader_info(db_connect):
    def leader_dict(leader_infos):
        leader_position_dict = {}
        for leader in leader_infos:
            leader_name = leader[1]
            if leader_name not in leader_position_dict.keys():
                leader_position_dict[leader_name] = []
            leader_organization = leader[4] if leader[4] is not None else ''
            leader_position_prefix = leader[3] if leader[3] is not None else ''
            leader_position = leader[2]

            leader_position_dict[leader_name].append(leader_organization
                                                     + leader_position_prefix
                                                     + leader_position)
        return leader_position_dict

    leader_infos = fetch_data_by_type(db_connect, 'leaders')
    leader_position_dict = leader_dict(leader_infos)

    return leader_position_dict


def merge_dict_assets(asset_a, asset_b):
    assets = []
    for item_a, item_b in zip(asset_a, asset_b):
        assets.append(set.union(item_a, item_b))
    return tuple(assets)


def post_process_sequence(input_sentence, model_json):
    inner_db_connect = sqlite3.connect('./database/inner_correct.db')
    outer_db_connect = sqlite3.connect('./database/outer_correct.db')

    inner_assets = get_db_dict_assets(inner_db_connect)
    outer_assets = get_db_dict_assets(outer_db_connect)
    leader_position_dict = build_leader_info(outer_db_connect)
    inner_db_connect.close()
    outer_db_connect.close()

    black_pairs, black_words, idioms, leader_name_set, leader_position_set, slogans = merge_dict_assets(inner_assets,
                                                                                                        outer_assets)

    model_json = post_disable_repeat(model_json)
    model_json = post_disable_empty_replace_text(model_json)
    model_json = post_disable_reverse_words(model_json)
    model_json = post_enable_mix_pair_symbol_detection(input_sentence, model_json)
    model_json = post_disable_bookname(input_sentence, model_json)
    model_json = post_disable_local(model_json)
    model_json = post_disable_number(model_json)
    model_json = post_disable_space_after_puncs(model_json)
    model_json = post_disable_rongyu(model_json)
    model_json = post_disable_space(model_json)
    model_json = post_disable_by_black_words(model_json, black_words)
    model_json = post_disable_by_black_pairs(model_json, black_pairs)
    model_json = post_disable_context(model_json)
    model_json = post_disable_remove_error_quotes(model_json)
    model_json = post_disable_gongwenguifan(model_json)
    model_json = post_disable_date(model_json)
    model_json = post_disable_error_quotation(input_sentence, model_json)
    model_json = post_disable_low(model_json)
    model_json = post_disable_nobody(input_sentence, model_json, leader_name_set, leader_position_set,
                                     leader_position_dict)
    model_json = post_disable_by_sentence_slogan(input_sentence, model_json, slogans)
    model_json = post_disable_idioms(model_json, idioms)
    # model_json = post_disable_symbol(input_sentence, model_json)
    model_json = correct_location_error(model_json, location_structure)
    model_json = post_enable_location(input_sentence, model_json, location_structure)
    return model_json
