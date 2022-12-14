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
        ??????????????????202??????????????????????????????????????????????????????
        ??????????????????????????????????????????????????????202?????????????????????
        :param model_json:
        :return:
        """
    p = re.compile(r"[\u4e00-\u9fa5]{1,4}[????????????]")
    p0 = re.compile(r"[\u4e00-\u9fa5]+???((?!???).)*???")
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
                #     item = createAdvise(text, start, '', '????????????')
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
                        item = createAdvise(text, start, '', '????????????')
                        sentence_error.append(item)

            if re.match(r"([\u4e00-\u9fa5]+???)?[\u4e00-\u9fa5]+???([\u4e00-\u9fa5]+[??????])?",text):
                exist_wrong, revise = wrong_gov_level(text,location_hashmap)
                if exist_wrong:
                    if not is_same_location_error_exist(text, sentence_error):
                        start = sentence.find(text)
                        item = createAdvise(text, start, revise,'???????????????')
                        sentence_error.append(item)

    return model_json

def delete_conjunctions(errorlist):
    res = []
    for error in errorlist:
        if exist_conjunctions(error):
            temp = re.split('???|???|???|???|???|???|???',error)
            for m in temp:
                res.append(m)
        else:
            res.append(error)
    return res

def exist_conjunctions(text):
    words = ['???','???','???','???','???','???','???']
    citys = ['???????????????','??????','??????','??????']
    for word in words:
        if word in text :
            for c in citys:
                if c in text:
                    return False
            return True
    return False

def clean_location_name(text,hashmap):
    p = re.compile(r"(?<=[????????????])")
    place = re.split(p, text)
    place.pop()
    if '??????' in place:
        place.remove('??????')
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
    ???????????????????????????????????????????????????????????????????????????????????????
    :param text:
    :param location_hashmap:
    :return: ??????????????????
    '''
    if text[-1] == '???':
        if text in location_hashmap['p2c']:
            return text
        else:
            for province_name in location_hashmap['p2c'].keys():
                if province_name in text:
                    text = province_name
                    return text
            return 'error'
    elif text[-1] == '???':
        if text in location_hashmap['c2co']:
            return text
        else:
            for city_name in location_hashmap['c2co'].keys():
                if city_name in text:
                    text = city_name
                    return text
            return 'error'
    elif text[-1] == '???':
        if text in location_hashmap['co2c']:
            return text
        else:
            for name in location_hashmap['co2c'].keys():
                if name in text:
                    text = name
                    return text
            return 'error'
    elif text[-1] == '???':
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
    ?????????????????????????????????????????????????????????
    :param text:
    :param location_hashmap:
    :return: bool???text
    '''
    p = re.compile(r"(?<=[????????????])")
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
        if place[0][-1] == '???':
            if place[1] in location_hashmap['p2c'][place[0]]:
                return False, ''
            else:
                if place[1] not in location_hashmap['c2p'] and place[1] in location_hashmap['c2co']:
                    return True, place[1]
                else:
                    maybep = location_hashmap['c2p'][place[1]][0]
                    return True, maybep + place[1]
        elif place[0][-1] == '???':
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
    ??????hash??????p2c???????????????,c2p???????????????,c2co??????????????????co2c???????????????
    :param location_structure:
    :return: hashmap{'p2c','c2p','c2co','co2c'}
    '''
    province_city = {}
    city_province = {}
    city_county = {}
    county_city = {}
    for province in location_structure:
        for city in province['city']:
            if city['name'] != '?????????' and city['name'] != '???':
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

            if city['name'] != '?????????' and city['name'] != '???':
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


def is_process_needed(text, location_structure):  # ??????
    p = re.compile(r"(?<=[??????])")
    place = re.split(p, text)  # ?????????????????????XX??????XX??????XX???
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
        'message': '??????????????????',
        'alertType': 4,
        'end': start + len(text) - 1,
        'errorType': 202,
        'replaceText': revise_202(text, location_structure),
        'sourceText': text,
        'start': start
    }
    return res


def revise_202(text, location_structure):  # ??????XX???XX????????????

    p = re.compile(r"(?<=[??????????????????])")
    place = re.split(p, text)  # ?????????????????????XX??????XX??????XX???
    place.pop()
    for province in location_structure:
        if province['province'] == place[0]:
            for city in province['city']:
                if place[1] in city["county"]:
                    return place[0] + city['name'] + place[1]
    return text


def post_disable_local(model_json):
    """
    ??????C1??????????????????????????????????????????????????????
    ????????????????????????????????????????????????????????????????????????????????????????????????????????????
    ???????????????????????????????????????????????????
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
    C6 ?????????????????????????????????????????????
    ????????????->"???????????????..."
    '''

    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            error_word = alert_error['replaceText']
            if alert_error['sourceText'][:2] == "??????" and error_word[:2] == "??????":
                print(alert_error['sourceText'][:2])
                continue
            elif error_word[:2] == "??????":
                continue
            elif alert_error['sourceText'][:2] == "??????" and error_word[:2] != "??????":
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_gongwenguifan(model_json):
    '''
    ?????????????????????????????????
    ?????????????????????????????????????????????
    ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    ???????????????????????????????????????????????????????????????????????????
    '''
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            error_word = alert_error['replaceText']
            error_type = alert_error['alertMessage']
            if error_type[0:8] == '????????????????????????':
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_space_after_puncs(model_json):
    '''
    "?????????????????????????????????,
    ????????????????????????????????????????????????,
    ?????????????????????????????????????????????
    '''
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if alert_error['errorType'] == 2 and alert_error['alertMessage'] == '????????????????????????':
                sentence_errors.remove(alert_error)
    return model_json


def post_disable_space(model_json):
    '''
    ????????????????????????????????????
    '''
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            error_cla = alert_error['sourceText']
            if len(error_cla) == 3:
                if error_cla[1] == ' ' or error_cla[1] == '??':
                    sentence_errors.remove(alert_error)
    return model_json


def post_disable_number(model_json):
    """
    ??????C12???????????????????????????????????????
    :param model_json:
    :return:
    """
    p = re.compile(r"[0-9]+[??????]+")  # ??????????????????????????????
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if re.match(p, alert_error['sourceText']):
                sentence_errors.remove(alert_error)
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            source_text = alert_error['sourceText']
            replace_text = alert_error['replaceText']
            if source_text[0] == '???' and source_text[-1] == '???' and replace_text[0] == '???' and replace_text[
                -1] == '???':
                source_number = int(source_text[1:-1])
                replace_number = cn2an.cn2an(replace_text[1:-1], "normal")
                if source_number == replace_number:
                    sentence_errors.remove(alert_error)
    return model_json


# def post_c1(model_json):
#     """
#     ??????C1????????????????????????????????????????????????
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
    ?????????????????????????????????
    :param model_json:
    :return:
    """
    len_book = 50
    for sentence, sentence_errors in zip(input_data['sentences'], model_json['alerts']):
        for alert_error in list(sentence_errors):
            start = alert_error['start']
            end = alert_error['end']
            sentence_range = sentence[max(0, start - len_book):min(len(sentence), end + len_book + 1)]
            res = re.finditer('???(.*?)???', sentence_range)
            for bookidx in res:
                start_idx, end_idx = bookidx.span()
                if max(0, start - len_book) + start_idx <= start and max(0, start - len_book) + end_idx > end:
                    sentence_errors.remove(alert_error)
                    break
    return model_json


def post_disable_date(model_json):
    """
    ??????????????????????????????
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
#     ????????????????????????????????????
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
#                 if last_word in '???????????????':
#                     sentence_errors.remove(alert_error)
#         model_json['alerts'][k] = [error for error in sentence_errors if error is not None]
#         k += 1
#
#     return model_json


def post_disable_rongyu(model_json):
    """
    ???????????????c13.1 ??? c7??????????????????
    ????????????
    :param model_json:
    :return:
    """
    target_string = ['??????', '???']  # ????????????alertMessage???????????????????????? ??? ??????????????????????????????????????????
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
        ?????????????????????????????????????????????
        :param model_json:input:??????
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

            if replaceText != "":  # ????????????????????????????????????
                new_item = creat_name_item(name, start, end, errorType, message, replaceText)
                for alert_error in sentence_errors:
                    if name in alert_error['sourceText'] and "???" in alert_error["sourceText"]:  # sourceText??????????????????
                        alert_error["alertMessage"] = alert_error["alertMessage"].replace(alert_error['sourceText'],
                                                                                          name)
                        alert_error["sourceText"] = name
                    elif name == alert_error['sourceText']:  # sourceText?????????????????????
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
        if position in string_before_name:  # ?????????????????????
            print(position)
            hit_position = position
            for leader_position in leader_position_list:
                if leader_position in string_before_name:  # ???????????????????????????
                    errorType = 667
                    message = "??????????????????????????????"
                    replaceText = name_o
                    return errorType, message, hit_position, replaceText
            if len(leader_position_list) > 0:  # ????????????????????????????????????
                errorType = 201
                message = "????????????????????????????????????:" + '???'.join(set(leader_position_list))
                replaceText = name_o
                return errorType, message, hit_position, replaceText
            elif len(leader_position_list) == 0:  # ????????????????????????????????????
                erroType = 201
                message = "????????????????????????"

                return erroType, message, hit_position, replaceText
    errorType = 667  # ??????????????????????????????
    message = "????????????????????????"
    replaceText = name_o
    return errorType, message, hit_position, replaceText


def post_disable_low(model_json):
    """
       ?????? ???????????????????????????????????????C12.4
       :param model_json:
       :return:
       """
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            message = alert_error['alertMessage']
            for j in range(0, len(message)):
                if message[j:] == "????????????????????????????????????????????????":
                    sentence_errors.remove(alert_error)
    return model_json


def post_disable_reverse_words(model_json):
    """
    ??????  ????????????C6.6
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
      ????????????
      :param model_json:input:??????
      :return:
      """

    def check_quotation(sentence, start_idx):
        return len(re.findall("???", sentence[0: start_idx])) % 2 == 0

    # ????????????sentence ???????????????
    for sentence, sentence_errors in zip(input_sentence["sentences"], model_json['alerts']):
        # ???????????????????????????
        # ????????????sentence??? error
        for alert_error in list(sentence_errors):
            if alert_error['alertMessage'] == "?????????????????????" and not check_quotation(sentence, alert_error["start"]):
                sentence_errors.remove(alert_error)

    return model_json


def post_disable_remove_error_quotes(model_json):
    for sentence_errors in model_json['alerts']:
        for alert_error in list(sentence_errors):
            if alert_error['sourceText'] == '???' + alert_error['replaceText'] + '???':
                sentence_errors.remove(alert_error)
    return model_json


def correct_location_error(model_json, location_structure):
    """
            ?????????????????????????????????????????????
            ???????????????source??? XX+?????????replace??? ?????????????????? ???????????? -> ???????????? ????????????
            :param model_json:
            :return:
            """
    pr = re.compile(r'[\u4e00-\u9fa5]{1,4}???$')
    p = re.compile(r'???')
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
    mix_pairs = {'???': ')', '(': '???', '???': ']', '[': '???', '???': '}', '{': '???', '???': '>', '<': '???'}
    english_to_chinese_pair = {"(": "???", ")": "???", "[": "???", "]": "???", "{": "???", "}": "???", "<": "???", ">": "???"}
    mix_pairs_inverse = {v: k for k, v in mix_pairs.items()}
    quotes_list = ['"', "'", '???', '???', '???', '???']

    for sentence, sentence_errors in zip(input_data['sentences'], model_json['alerts']):
        for alert_error in list(sentence_errors):
            if '??????' in alert_error['alertMessage']:
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
                        alert_error['message'] = '???????????????????????????'
                        alert_error['alertType'] = 10
                elif alert_error['sourceText'] in quotes_list:
                    alert_error['message'] = '???????????????????????????'
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
    black_pairs = fetch_data_by_type(db_connect, 'black_pairs')#????????????
    black_pairs = set([pair[1] + ' ' + pair[2] for pair in black_pairs])

    black_words = fetch_data_by_type(db_connect, 'black_words')#??????source
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
