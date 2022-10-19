from post_process import post_process_sequence

error_map = {
    1: 1,
    2: 2,
    3: 3,
    5: 4,
    6: 5,
    101: 6,
    102: 7,
    103: 9,
    105: 1,
    201: 9,
    202: 10,
    666: 11,
    667: 12
}

operatorType = {
    0: '无需操作',
    1: '向前新增',
    2: '向后新增',
    3: '删除',
    4: '替换',
    16: '无需操作',
    666: '修改',
    10: '谨慎查验'
}


def deal_with_alert(fixed):
    output_list = []
    for sentence_alerts in fixed:
        output_alert_list = []
        if len(sentence_alerts) > 0:
            for alert in sentence_alerts:
                output_alert = {}
                if alert['errorType'] == 105:
                    continue

                output_alert['suggestion'] = operatorType[alert['alertType']]
                if 'message' in alert:
                    output_alert['message'] = alert['message']

                if 'replaceText' in alert:
                    output_alert['fixed'] = alert['replaceText']
                else:
                    output_alert['suggestion'] = operatorType[666]
                output_alert['source'] = alert['sourceText']
                output_alert['start_index'] = alert['start']
                output_alert['end_index'] = alert['end']
                output_alert['advise'] = alert['advancedTip']
                output_alert['error_type'] = error_map[alert['errorType']]
                output_alert_list.append(output_alert)
        output_list.append(output_alert_list)
    return output_list


if __name__ == '__main__':
    data = {}
    data_json = data['data_json']
    result_json = data['result_json']
    result = post_process_sequence(data_json, result_json)
    alerts = result['alerts']

    output = deal_with_alert(alerts)
    response = {'detected_errors': output}
    print(response)
