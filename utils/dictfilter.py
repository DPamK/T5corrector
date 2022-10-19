import os
from loguru import logger

class dictfiliter():
    def __init__(self,dictpath='blank_words') -> None:
        files = os.listdir(dictpath)
        # 对于需要过滤的dict分为四类，其中固定名词1、替换2、插入3、删除4
        self.nameDict = []
        self.replaceDict = {}
        self.insertDict = {}
        self.deleteDict = []
        # 读取字典文件，保存在对应的dict里面，初始化需要时间，所以不要频繁实例化
        for file in files:
            type = file.split('.')[-1]
            if type != 'txt':
                continue
            filename = file.split('.')[0]
            dictkind = filename.split('_')[-1]
            if dictkind == '1':
                path = os.path.join(dictpath,file)
                with open(path,'r',encoding='utf8') as f:
                    words = f.readlines()
                for word in words:
                    word = word.strip()
                    self.nameDict.append(word)
            elif dictkind == '2':
                path = os.path.join(dictpath,file)
                with open(path,'r',encoding='utf8') as f:
                    pairs = f.readlines()
                for pair in pairs:
                    pair = pair.strip()
                    ori,fix = pair.split('->')
                    if ori in self.replaceDict:
                        self.replaceDict[ori].append(fix)
                    else:
                        self.replaceDict[ori] = [fix]
            elif dictkind == '3':
                path = os.path.join(dictpath,file)
                with open(path,'r',encoding='utf8') as f:
                    pairs = f.readlines()
                for pair in pairs:
                    pair = pair.strip()
                    ori,fix = pair.split('->')
                    if ori in self.insertDict:
                        self.insertDict[ori].append(fix)
                    else:
                        self.insertDict[ori] = [fix]
            elif dictkind == '4':
                path = os.path.join(dictpath,file)
                with open(path,'r',encoding='utf8') as f:
                    words = f.readlines()
                for word in words:
                    word = word.strip()
                    self.deleteDict.append(word)
            else:
                continue
        logger.info('dict filiter init finish.')
        logger.info(f'namedict:{len(self.nameDict)} replacedict:{len(self.replaceDict)} insertdict:{len(self.insertDict)} deletedict:{len(self.deleteDict)}')

    def filterName(self,alert):
        source = alert['sourceText']
        return source in self.nameDict

    def filterReplace(self,alert):
        source = alert['sourceText']
        fixed = alert['replaceText']
        if source in self.replaceDict:
            return fixed in self.replaceDict[source]
        else:
            return False


    def filiterInsert(self,alert):
        source = alert['sourceText']
        fixed = alert['replaceText']
        if source in self.insertDict:
            return fixed in self.insertDict[source]
        else:
            return False

    def filterDelete(self,alert):
        source = alert['sourceText']
        return source in self.deleteDict

    def filter(self,alert):
        kind = alert['alertType']
        # alert 中 alertType : 4 为替换replace； 3为删除delete； 2为添加insert
        if self.filterName(alert):
            return True

        if kind == 4:
            return self.filterReplace(alert)
        elif kind == 3:
            return self.filterDelete(alert)
        elif kind == 2:
            return self.filiterInsert(alert)
        else:
            return False


if __name__=='__main__':
    alertlist = [
        {
            "advancedTip": "True",
            "alertMessage": "建议删除“检史”",
            "alertType": 3,
            "errorType": 5,
            "replaceText": "",
            "sourceText": "检史",
            "start": 630,
            "end": 632
        },
        {
            "advancedTip": "False",
            "alertMessage": "建议用“会议”替换“会”",
            "alertType": 4,
            "errorType": 1,
            "replaceText": "会议",
            "sourceText": "会",
            "start": 1125,
            "end": 1126
        },
        {
            "advancedTip": "True",
            "alertMessage": "建议用“闽学”替换“敏学”",
            "alertType": 4,
            "errorType": 1,
            "replaceText": "闽学",
            "sourceText": "敏学",
            "start": 1180,
            "end": 1182
        },
        {
            "advancedTip": "True",
            "alertMessage": "建议添加“教育”",
            "alertType": 2,
            "errorType": 5,
            "replaceText": "教育",
            "sourceText": "学习",
            "start": 1285,
            "end": 1287
        },
        {   
            "advancedTip": "True",
            "alertMessage": "建议用“车仕安”替换“辉仕安”",
            "alertType": 4,
            "errorType": 1,
            "replaceText": "车仕安",
            "sourceText": "辉仕安",
            "start": 1830,
            "end": 1833
        }
    ]

    filter = dictfiliter()
    res = []
    for alert in alertlist:
        if filter.filter(alert):
            continue
        else:
            res.append(alert)
    print(res)