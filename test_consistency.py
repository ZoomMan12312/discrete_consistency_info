import pandas as pd
import numpy as np
import math

def unit_test():
    data = pd.read_csv('mushrooms.csv')
    y = data['class']
    y_name = y.name
    y_vector = y.values
    
    x = data.drop(['class'], axis=1)
    x_colNames = x.columns.values
    x_values = x.values
    x_vectors = x_values.transpose()

    ci = ConsistencyInfo(x_vectors, x_colNames, y_vector)
    top_feature = ci.top_feature
    y_uniq = ci.y_uniq
    x_uniq = ci.x_uniq

    '''
    print(x_vectors[top_feature[2]])
    unique, counts = np.unique(x_vectors[top_feature[2]], return_counts=True)
    print(dict(zip(unique, counts)))
    for y in y_uniq:
        for x in x_uniq[top_feature[2]]:
            i = 0
            cnt = 0
            for x_iter in x_vectors[top_feature[2]]:
                if x_iter == x and y_vector[i] == y:
                    cnt += 1
                i+=1
            print('y: ' + str(y) + ', x:' + str(x) + ', cnt:' + str(cnt))
    '''
    print(ci.sorted_pairs)

class ConsistencyInfo():
    def __init__(self, x_vectors, x_colNames, y_vector, tol=15):
        self.tol = tol
        self.x_vectors = x_vectors
        self.x_colNames = x_colNames
        self.y_vector = y_vector
        self.y_uniq = np.unique(y_vector)
        self.n = len(self.y_uniq)
        self.y_share = 1/self.n
        self._setVectorUniqueItems()
        self._getAllProbas()
        self._getAllConsistency()
        self._getAllFeatureConsistency()
        self._getTopFeature()
        #print(self.x_uniq)
        

    def _setVectorUniqueItems(self):
        self.x_uniq = []
        for i in self.x_vectors:
            self.x_uniq.append(np.unique(i))

    def _getAllProbas(self):
        i = 0
        self.x_probas = []
        for vector in self.x_vectors:
            self.x_probas.append([])
            for attribute in self.x_uniq[i]:
                res = self._getProba(vector, attribute)
                self.x_probas[i].append(res)
            i += 1
        #print(self.x_probas)
    
    def _getProba(self, vector, attribute):
        probas = []
        probaSum = 0
        #print(attribute)
        for y in self.y_uniq:
            #print(y)
            match = 0
            cnt = 0
            for x in vector:
                if x == attribute and self.y_vector[cnt] == y:
                    match += 1
                cnt += 1
            proba = (match/cnt)*self.y_share
            #print(proba)
            probaSum += proba
            probas.append(proba)
        normalized = []
        for i in probas:
            normalized.append(i/probaSum)
            #print(i/probaSum)
        #print('--------------------------------')
        return normalized

    def _getAllConsistency(self):
        i = 0
        self.x_consistency = []
        for proba in self.x_probas:
            self.x_consistency.append([])
            for x in proba:
                res = self._getConsistency(x)
                self.x_consistency[i].append(res)
            i += 1
        #print(self.x_consistency)

    def _getConsistency(self, probas):
        cs = []
        for proba in probas:
            if proba <= self.y_share:
                c = 1 - ( 1/( 1+math.exp( -10*self.n*proba + 5 ) ) )
            else:
                c = 1/( 1+math.exp( (5*self.n*(-2*proba + 1) +5) ) )
            cs.append(c)
        p = 1
        for c in cs:
            p *= c
        return p

    def _getAllFeatureConsistency(self):
        self.feature_consistency = []
        for cs in self.x_consistency:
            fc = self._getFeatureConsistency(cs)
            self.feature_consistency.append(fc)
        
    
    def _getFeatureConsistency(self, cs):
        m = len(cs)
        fc_mean = 0
        fc_max = 0
        fc_share = 1/m
        for c in cs:
            fc_mean += c/m
            if c > fc_max:
                fc_max = c
        fc_score = ((fc_max + fc_mean)/2)*( (2*(fc_share**(1/self.tol))) - (2*(0.5**(1/self.tol))) + 1)
        return fc_score

    def _getTopFeature(self):
        i = 0
        self.pairs = []
        self.top_feature = [0,'', 0]
        for f in self.feature_consistency:
            if f > self.top_feature[0]:
                self.top_feature = [f, self.x_colNames[i], i]
            self.pairs.append([f, self.x_colNames[i], i])
            i += 1
        self.sorted_pairs = sorted(self.pairs, key=lambda x: x[0])
        #print(self.pairs)
        #print(self.top_feature)


unit_test()