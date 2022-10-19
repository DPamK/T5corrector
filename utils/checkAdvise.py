import synonyms
import numpy as np

def SimComputer(src, fixed, t=0.35):
    try:
        vec1 = synonyms.v(src)
        vec2 = synonyms.v(fixed)
        cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if cos_sim > t:
            return (True,True)
        else:
            return (True,False)
    except:
        print('不在字典！错误！')
        cos_sim = -1
        return (False,True)