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
        self.S = ['one', 'two', 'three']
        self.K = ['red', 'white']
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
    
    def generate_data(self, times):
        def _generate_data_by_P(P):
            return np.where(np.random.multinomial(1, P) == 1)[0][0]
        
        #init
        S = []
        O = []
        S.append(self.S[_generate_data_by_P(list(self.PI.values()))])
        O.append(self.K[_generate_data_by_P(list(self.B[S[0]].values()))])
        for time in range(1, times):
            S.append(self.S[_generate_data_by_P(list(self.A[S[time-1]].values()))])
            O.append(self.K[_generate_data_by_P(list(self.B[S[time]].values()))])
        return S, O
    
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
    
    def study_unsupervise(self, O, diff=0.002):
        #init
        self.PI = {}
        temp_list = self.generate(len(self.S))
        oldPI = temp_list
        [self.PI.update(dictvaule) for dictvaule in [{sub:temp_list[index]} for index, sub in enumerate(self.S)]]
        
        self.A = {}
        oldA = []
        for state in self.S:
            temp_list = self.generate(len(self.S))
            self.A[state] = {}
            oldA.append(temp_list)
            [self.A[state].update(dictvaule) for dictvaule in [{sub:temp_list[index]} for index, sub in enumerate(self.S)]]
        
        self.B = {}
        oldB = []
        for state in self.S:
            temp_list = self.generate(len(self.K))
            self.B[state] = {}
            oldB.append(temp_list)
            [self.B[state].update(dictvaule) for dictvaule in [{sub:temp_list[index]} for index, sub in enumerate(self.K)]]
        
        print(self.PI)
        print(self.A)
        print(self.B)
        
        #为退出做准备
        count = 0
        while True:
            self.evaluate_probability(O, method='forward')
            self.evaluate_probability(O, method='backward')
            
            #P(q_t=S_t,q_t+1=S_t+1 | O,λ)
            #t=0:S_t[self.S] S_t+1[self.S] ==> len(self.S) * len(self.S)
            #t=0...T-1:T-1 * N * N
            P_union = np.zeros((len(O)-1, len(self.S), len(self.S)))
            for time in range(len(O)-1):
                for st_index, st in enumerate(self.S):
                    for st_1_index, st_1 in enumerate(self.S):
                        P_union[time][st_index][st_1_index] = self.forward[time][st_index] * self.A[st][st_1] * self.B[st_1][O[time+1]] * self.backward[time+1][st_1_index]
                #还有一个除的操作(归一化)
                P_union[time] / P_union[time].sum()
            #print(P_union)
                #数组简化代码
                #norm = self.forward[time, :] * self.A * self.B[:, O[time+1]] * self.backward[time+1, :]
            
            #gamma ==> 就是st_1固定,st取值不定
            gamma = np.sum(P_union, axis=2)#只有T-1,还有T状态
            #(alpha[T-1,:] * beta[T-1,:])要多思考
            temp = []
            for sT_index, sT in enumerate(self.S):
                temp.append(self.forward[-1][sT_index] * self.backward[-1][sT_index])
            temp = np.array(temp) / np.array(temp).sum()
            gamma = np.vstack((gamma, temp))
            #print(gamma)
            
            newPI = gamma[0,:]
            newA = []
            for st_index, st in enumerate(self.S):
                lower = [gamma[time][st_index] for time in range(len(O)-1)]
                newA.append([P_union[:, st_index, st_1_index].sum() / sum(lower) for st_1_index in range(len(self.S))])
            newB = []
            for st_index, st in enumerate(self.S):
                lower = [gamma[time][st_index] for time in range(len(O))]
                newB.append([sum([gamma[time][st_index] for time, O_val in enumerate(O) if kt == O_val]) / sum(lower) for kt in self.K])
            
            #print(newPI)
            #print(newA)
            #print(newB)
            
            #结束条件 迭代次数 + 收敛
            if count > 1000:
                print('count over')
                break
            else:
                count += 1
            
            differ = [] 
            differ.append(np.fabs(np.array(newPI) - np.array(oldPI)).max())
            differ.append(np.fabs(np.array(newA) - np.array(oldA)).max())
            differ.append(np.fabs(np.array(newB) - np.array(oldB)).max())
            if max(differ) < diff:
                print(max(differ))
                break
            
            oldPI = newPI
            oldA = newA
            oldB = newB
            
            #update
            [self.PI.update(dictvaule) for dictvaule in [{sub:newPI[index]} for index, sub in enumerate(self.S)]]
            for s_index, state in enumerate(self.S):
                [self.A[state].update(dictvaule) for dictvaule in [{sub:newA[s_index][index]} for index, sub in enumerate(self.S)]]
        
            for s_index, state in enumerate(self.S):
                [self.B[state].update(dictvaule) for dictvaule in [{sub:newB[s_index][index]} for index, sub in enumerate(self.K)]]
            
        print(self.PI)
        print(self.A)
        print(self.B)
    
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
    S, O = hmm_test.generate_data(5)
    hmm_test.study_unsupervise(O)
    
#    print(hmm_test.evaluate_probability(['red', 'white', 'red'], method='backward', steptime=2))
#    print(hmm_test.pre_predict(['red', 'white', 'red']))
#    https://www.cnblogs.com/skyme/p/4651331.html    