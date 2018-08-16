# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:28:29 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import numpy as np

class hmm():
    def __init__(self):
        #这用来验证评估和预测的问题
#        self.S = ('Healthy', 'Fever')
#        self.K = ('normal', 'cold', 'dizzy')
#        self.PI = {'Healthy':0.8, 'Fever':0.2}
#        self.A = {
#                'Healthy':{'Healthy':0.6, 'Fever':0.4},
#                'Fever':{'Healthy':0.3, 'Fever':0.7}
#                }
#        self.B = {
#                'Healthy':{'normal': 0.6, 'cold': 0.2, 'dizzy': 0.2},
#                'Fever':{'normal': 0.1, 'cold': 0.5, 'dizzy': 0.4}
#                }
#        
        self.S = ('one', 'two', 'three')
        self.K = ('red', 'white')
        self.PI = {'one':0.2, 'two':0.4, 'three':0.4}
        self.A = {
                'one':{'one':0.5, 'two':0.2, 'three':0.3},
                'two':{'one':0.3, 'two':0.5, 'three':0.2},
                'three':{'one':0.2, 'two':0.3, 'three':0.5}
                }
        self.B = {
                'one':{'red':0.5, 'white':0.5},
                'two':{'red':0.4, 'white':0.6},
                'three':{'red':0.7, 'white':0.3}
                }
        
        self.forward = []
        self.backward = []
        
        self.predict_value = []
        self.predict_state = []
    
    #question 1
    def evaluate_probability_forward(self, O, times, endpos=0):
        if 0 == times:
            self.forward.append([self.PI[value] * self.B[value][O[times]] for value in self.S])
        else:
            temp_list = []
            for index, value in enumerate(self.S):
                temp = 0
                for oldindex, oldvalue in enumerate(self.forward[times-1]):
                    temp += oldvalue * self.A[self.S[oldindex]][value]
                temp = temp * self.B[self.S[index]][O[times]]
                temp_list.append(temp)
            self.forward.append(temp_list)
        if endpos > (times + 1):
            self.evaluate_probability_forward(O, times + 1, endpos)
    
    def evaluate_probability_backward(self, O, times, endpos=0):
        if self.backward:
            temp_list = []
            for index, value in enumerate(self.S):
                temp = 0
                for oldindex, oldvalue in enumerate(self.backward[len(O) - times - 1]):
                    temp += self.A[value][self.S[oldindex]] * self.B[self.S[oldindex]][O[times]] * oldvalue
                temp_list.append(temp)
            self.backward.append(temp_list)
        else:
           self.backward.append([1 for index in range(len(self.S))])
        if endpos < (times - 1):
           self.evaluate_probability_backward(O, times - 1, endpos)
    
    def evaluate_probability(self, O, method='forward', steptime=0):
        if 'forward' == method:
            self.forward.clear()
            self.evaluate_probability_forward(O, times=0, endpos=len(O))
            #print(self.forward)
            #print(sum(self.forward[len(O)-1]))
            return sum(self.forward[len(O)-1])
        elif 'backward' == method:
            self.backward.clear()
            self.evaluate_probability_backward(O, times=len(O))
            #print(self.backward)
            #print(sum([self.PI[value] * self.B[value][O[0]] * self.backward[-1][index] for index, value in enumerate(self.S)]))
            return sum([self.PI[value] * self.B[value][O[0]] * self.backward[-1][index] for index, value in enumerate(self.S)])
        elif 'mix' == method:
            if steptime in range(len(O)):
                self.evaluate_probability_forward(O, times=0, endpos=steptime)
                self.evaluate_probability_backward(O, times=len(O), endpos=steptime-1)
                #print(self.forward)
                #print(self.backward)
                #print(sum([self.forward[-1][index] * self.backward[-1][index] for index, value in enumerate(self.S)]))
                return sum([self.forward[-1][index] * self.backward[-1][index] for index, value in enumerate(self.S)])
    
    def generate(self, count):
        generate_list = [np.random.random()]
        for index in range(count):
            if len(generate_list) == (count-1 ):
                generate_list.append(1-sum(generate_list))
                break
            while True:
                temp = np.random.random()
                if temp < (1-sum(generate_list)):
                    generate_list.append(temp)
                    break
        return generate_list
    
    def study_unsupervise(self, O):
        #init
        self.PI = {'one':0.2, 'two':0.4, 'three':0.4}
        self.PI = {}
        temp_list = self.generate(len(self.S))
        [self.PI.update(dictvaule) for dictvaule in [{sub:temp_list[index]} for index, sub in enumerate(self.S)]]
        
        self.A = {}
        for state in self.S:
            temp_list = self.generate(len(self.S))
            self.A[state] = {}
            [self.A[state].update(dictvaule) for dictvaule in [{sub:temp_list[index]} for index, sub in enumerate(self.S)]]
        
        self.B = {}
        for state in self.S:
            temp_list = self.generate(len(self.K))
            self.B[state] = {}
            [self.B[state].update(dictvaule) for dictvaule in [{sub:temp_list[index]} for index, sub in enumerate(self.K)]]
        
#        print(self.PI)
#        print(self.A)
#        print(self.B)
        
        #EM算法还是没有理解透彻,还是没有搞清楚,不知道要怎么计算,等我后期再看看来补上
        #E_step求联合概率的期望
        #expect = np.zeros([len(O), len(self.S)])
        #for index, value in enumerate(O):
            #P(Z|X)=P(Z)*P(X|Z) / P(X)
            #x_given_z = [self.B[state][value] * self.PI[state] for state in self.S]
            #x_given_z = np.array(x_given_z) / sum(x_given_z)
            
                
        #M_step最大化
    
    def study_supervise(self):#这个需要大量数据
        self.S = ('end', 'not end')
        self.K = ('杨', '娜', '方')
        String = '杨娜 杨树 杨家 老杨 老杨树 老杨家 婀娜多姿 娜娜 娜爷 袅袅娜娜 杨枝 娜 杨 四方 方圆 地方 双方 方桌 方'
        words = []
        tag = []
        
        for index, string in enumerate(String):            
            if ' ' == string:
                tag[-1] = 1
            else:
                words.append(string)
                tag.append(0)
        tag[-1] = 1
        
        #PI
        self.PI = [tag.count(index) for index, value in enumerate(self.S)]
        
        #A
        count_00, count_01, count_10, count_11 = 0, 0, 0, 0
        for index in range(len(tag)-1):
            if 0 == tag[index]:
                if 0 == tag[index+1]:
                    count_00 += 1
                elif 1 == tag[index+1]:
                    count_01 += 1
            elif 1 == tag[index]:
                if 0 == tag[index+1]:
                    count_10 += 1
                elif 1 == tag[index+1]:
                    count_11 += 1
        #print(count_00, count_01, count_10, count_11)
        self.A = {}
        self.A[self.PI[0]] = {self.PI[0]:count_00/(count_00+count_01), self.PI[1]:count_01/(count_00+count_01)}
        self.A[self.PI[1]] = {self.PI[0]:count_10/(count_10+count_11), self.PI[1]:count_11/(count_10+count_11)}
        
        #B
        count = {}
        for index, word in enumerate(words):
            if word in self.K:
                if word not in count:
                    temp = [0, 0]
                    temp[tag[index]] += 1
                    count[word] = temp
                else:
                    count[word][tag[index]] += 1
        results = [(key, item / sum(value)) for key, value in count.items() for item in value]
        self.B = {}
        [self.B.update(updatedict) for updatedict in [{state:{}} for state in self.S]]
        for index, result in enumerate(results):
            real_index = index % len(self.S)
            self.B[self.S[real_index]][result[0]] = result[1]
        #print(self.B)
        
        return self.PI, self.A, self.B

    def predict(self, O, times):
        if 0 == times:
            self.predict_value.clear()
            self.predict_state.clear()
            self.predict_value.append([self.PI[value] * self.B[value][O[times]] for value in self.S])
            self.predict_state.append([0 for value in self.S])
        else:
            value_list = []
            state_list = []
            for index, value in enumerate(self.S):
                maxnum = 0
                maxidx = 0
                for oldindex, oldvalue in enumerate(self.predict_value[times-1]):
                    current_num = oldvalue * self.A[self.S[oldindex]][value] * self.B[value][O[times]]
                    if current_num > maxnum:
                        maxnum = current_num
                        maxidx = oldindex
                value_list.append(maxnum)
                state_list.append(maxidx)
            self.predict_value.append(value_list)
            self.predict_state.append(state_list)
        
        if len(O) > (times + 1):
            self.predict(O, times + 1)
    
    def pre_predict(self, O):
        self.predict(O, times=0)
        trace = [np.array(self.predict_value[-1]).argmax()]
        for index in range(len(O)-1):
            trace.append(self.predict_state[-(index+1)][trace[index]])
        #print(self.predict_value)
        #print(self.predict_state)
        #print(max(self.predict_value[-1]))
        return max(self.predict_value[-1]), trace


if __name__ == '__main__':
    hmm_test = hmm()
    hmm_test.study_unsupervise(['red', 'white', 'red'])
#    print(hmm_test.evaluate_probability(['red', 'white', 'red'], method='backward', steptime=2))
#    print(hmm_test.pre_predict(['red', 'white', 'red']))
#    https://www.cnblogs.com/skyme/p/4651331.html    