from collections import defaultdict
import re



class NaiveFilter():
    def __init__(self):
        self.keywords = set([])

    def parse(self, path):
        for keyword in open(path):
            self.keywords.add(keyword.strip().decode('utf-8').lower())

    def filter(self, message, repl="*"):
        message = str(message).lower()
        for kw in self.keywords:
            message = message.replace(kw, repl)
        return message


class BSFilter:
    def __init__(self):
        self.keywords = []
        self.kwsets = set([])
        self.bsdict = defaultdict(set)
        self.pat_en = re.compile(r'^[0-9a-zA-Z]+$')  # english phrase or not

    def add(self, keyword):
        if not isinstance(keyword, str):
            keyword = keyword.decode('utf-8')
        keyword = keyword.lower()
        if keyword not in self.kwsets:
            self.keywords.append(keyword)
            self.kwsets.add(keyword)
            index = len(self.keywords) - 1
            for word in keyword.split():
                if self.pat_en.search(word):
                    self.bsdict[word].add(index)
                else:
                    for char in word:
                        self.bsdict[char].add(index)

    def parse(self, path):
        with open(path, "r") as f:
            for keyword in f:
                self.add(keyword.strip())

    def filter(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        message = message.lower()
        for word in message.split():
            if self.pat_en.search(word):
                for index in self.bsdict[word]:
                    message = message.replace(self.keywords[index], repl)
            else:
                for char in word:
                    for index in self.bsdict[char]:
                        message = message.replace(self.keywords[index], repl)
        return message


class DFAFilter():
    def __init__(self):
        self.keyword_chains = {}
        self.delimit = '\x00'

    def add(self, keyword):
        if not isinstance(keyword, str):
            keyword = keyword.decode('utf-8')
        keyword = keyword.lower()
        chars = keyword.strip()
        if not chars:
            return
        level = self.keyword_chains
        for i in range(len(chars)):
            if chars[i] in level:
                level = level[chars[i]]
            else:
                if not isinstance(level, dict):
                    break
                for j in range(i, len(chars)):
                    level[chars[j]] = {}
                    last_level, last_char = level, chars[j]
                    level = level[chars[j]]
                last_level[last_char] = {self.delimit: 0}
                break
        if i == len(chars) - 1:
            level[self.delimit] = 0

    def parse(self, path):
        with open(path) as f:
            for keyword in f:
                self.add(keyword.strip())

    def filter(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        message = message.lower()
        ret = []
        res = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for char in message[start:]:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        ret.append(repl * step_ins)
                        res.append(message[start:start + step_ins])
                        start += step_ins - 1
                        break
                else:
                    ret.append(message[start])
                    break
            else:
                ret.append(message[start])
            start += 1

        return res, ''.join(ret)

def filter_SensitiveWord(sentence):
    gfw = DFAFilter()
    gfw.parse("new_keywords.txt")
    wordList ,_ = gfw.filter(sentence)
    wrongmessage = []
    for word in wordList:
        start = sentence.find(word)
        end = start + len(word) +1
        temp = {
            'advancedTip':True,
            'alertMessage':f'建议修改"{word}"',
            'alertType':99,
            'errorType':1,
            'replaceText':"",
            'sourceText':word,
            'start':start,
            'end':end
        }
        wrongmessage.append(temp)
    return wrongmessage

# gfw = DFAFilter()
# gfw.parse("new_keywords.txt")
# 将后处理的json修改到原文
# def exe_filter(wrongMessage,text):
#     right = text
#     for wrong in wrongMessage:
#         start = wrong['start']
#         end = wrong ['end'] -1
#         replace = wrong['replaceText']
#         right = right[:start] + replace + right[end:]
#     return right

# 针对输出结果进行敏感词审查
# def filter_SensitiveWord(sentence):
#     wordList ,_ = gfw.filter(sentence)
#     wrongmessage = []
#     for word in wordList:
#         start = sentence.find(word)
#         if start != -1:
#             end = start + len(word) +1
#             temp = {
#                 'replaceText':"*"*len(word),
#                 'sourceText':word,
#                 'start':start,
#                 'end':end
#             }
#             wrongmessage.append(temp)
#     return wrongmessage


if __name__ == "__main__":
    sentence = "法伦功可以保持飞跃征战，tmd我操"
    # gfw = DFAFilter()
    # gfw.parse("new_keywords.txt")
    # import time
    # t = time.time()
    # wordList ,_ = gfw.filter("法伦功可以保持飞跃征战，tmd我操", "*")
    # for word in wordList:
    #     print("建议修改“"+word+'”')
    # i = 0
    # with open("output.txt", 'w') as fw:
    #     with open("zz.txt") as f:
    #         for keyword in f:
    #             wordList ,_ = gfw.filter(keyword, "*")
    #             for word in wordList:
    #                 fw.write(word+'\n')
    #             #print(i, gfw.filter(keyword, "*"))
    #             i += 1

    # s1 = set(open('new_keywords.txt','r').readlines())
    # s2 = set(open('output.txt','r').readlines())
    # dif = s1.difference(s2)

    # with open("new_keywords.txt", 'w') as fw:
    #     for word in dif:
    #         fw.write(word)
    # print(time.time() - t)