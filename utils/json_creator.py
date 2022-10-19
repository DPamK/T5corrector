
from utils.checkAdvise import SimComputer

def judgeadvice(sourceText,replaceText):
    frequent, similer = SimComputer(sourceText,replaceText)
    if frequent:
        advancedTip = str(similer)
    else:
        advancedTip = str(similer)
    return advancedTip

def creat_alert_item(alertMessage, alertType, errorType, replaceText, sourceText, ori_pos_begin, ori_pos_end):
    if alertType == 4:
        advancedTip = judgeadvice(sourceText,replaceText)
    elif alertType == 1:
        advancedTip = judgeadvice(sourceText,replaceText+sourceText)
    elif alertType == 2:
        advancedTip = judgeadvice(sourceText,sourceText+replaceText)
    elif alertType == 3:
        advancedTip = str(True)
    else:
        advancedTip = str(True)
    
    alert_item = item(advancedTip, alertMessage, alertType, errorType, replaceText, sourceText,
                                    ori_pos_begin, ori_pos_end)
    return  alert_item

def item(advancedTip, alertMessage, alertType, errorType, replaceText, sourceText, ori_pos_begin, ori_pos_end):
    res = {
        'advancedTip': advancedTip,
        'alertMessage': alertMessage,
        'alertType': alertType,
        'errorType': errorType,
        'replaceText': replaceText,
        'sourceText': sourceText,
        'start': ori_pos_begin,
        'end': ori_pos_end,
    }
    return res