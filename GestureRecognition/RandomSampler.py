'''
Created on Dec 15, 2016

@author: hshi
'''
from Sampler import Sampler
import numpy as np
import random as rd
class RandomSampler(Sampler):
    '''
    classdocs
    '''


    def __init__(self,
                 sampleNames_all = None,
                 ratio_train = 0.64,
                 ratio_valid = 0.18,
                 ratio_test = 0.18):
        '''
        Constructor
        '''
        super(RandomSampler, self).__init__(sampleNames_all, ratio_train, ratio_valid)
        self.ratio_test = ratio_test
        
        
        
        
        if self.sampleNames_all !=None:
            self.sampleNames_all = self.shuffleSampleNames_all(sampleNames_all)
            self.sampleNum = len(self.sampleNames_all)
            
            self.sampleNum_test = np.ceil(self.sampleNum * self.ratio_test).astype('int')
            self.sampleNum_valid = np.floor(self.sampleNum * self.ratio_valid).astype('int')
            
            self.sampleNum_train = self.sampleNum - self.sampleNum_valid - self.sampleNum_test
            
            self.sampleInd = np.linspace(0, self.sampleNum - 1, self.sampleNum).astype('int')
        
    def shuffleSampleNames_all(self, sampleNames):
        
        sampleNum = len(sampleNames)
        sampleInd = np.linspace(0, sampleNum - 1, sampleNum).astype('int')
        rd.shuffle(sampleInd)
        
        destSampleNames = list()
        
        for i in range(sampleNum):
            destSampleNames.append(sampleNames[sampleInd[i]])
            
        return destSampleNames
        
    def getRatio_test(self):
        return self.ratio_test
    
    
    #def getNextSamplingResult(self):
        