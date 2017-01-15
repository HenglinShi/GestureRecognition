'''
Created on Dec 15, 2016

@author: hshi
'''


import numpy as np
from RandomSampler import RandomSampler
from CrossValidatingRandomSamplingResult import CrossValidatingRandomSamplingResult

class CrossValidatingRandomSampler(RandomSampler):
    '''
    classdocs
    '''


    def __init__(self,
                 sampleNames_all = None,
                 ratio_train = 0.64,
                 ratio_valid = 0.18,
                 n_folds = 5):
        '''
        Constructor
        '''
        ratio_test = 1.0/n_folds
        super(CrossValidatingRandomSampler, self).__init__(sampleNames_all, ratio_train, ratio_valid, ratio_test)
        #self.sampleNames_all = sampleNames_all

        self.n_folds =n_folds
        
        self.cvIndex = 0
        


        
        
    def getNextSamplingResult(self):
    
        if self.cvIndex < self.n_folds:

            sampleIndBeg_test = self.cvIndex * self.sampleNum_test
            
            self.cvIndex += 1
            
            sampleIndEnd_test = self.cvIndex * self.sampleNum_test
            
            if sampleIndEnd_test > self.sampleNum:
                sampleIndEnd_test = self.sampleNum
            
            sampleNames_test = self.sampleNames_all[sampleIndBeg_test:sampleIndEnd_test]   
                
            sampleNames_train_valid = np.concatenate((self.sampleNames_all[0:sampleIndBeg_test],
                                                      self.sampleNames_all[sampleIndEnd_test:self.sampleNum]),
                                                     axis = 0)
    
            # Ramdonly split 
            sampleNames_train, sampleNames_valid = self.RadomSplitting_train_valid(sampleNames_train_valid)
                                    
            
            return CrossValidatingRandomSamplingResult(sampleNames_train = sampleNames_train,
                                                       sampleNames_valid = sampleNames_valid,
                                                       sampleNames_test = sampleNames_test,
                                                       foldIte = self.cvIndex)
                                                       
        
        else:
            return None
     
        
        #return sampleNames_train, sampleNames_valid, sampleNames_test
    
    
        
    def RadomSplitting_train_valid(self, sampleNames_train_valid):
        
        sampleNum = len(sampleNames_train_valid)
        
        #sampleNum_valid = np.floor(sampleNum * self.ratio_valid)
        #sampleNum_train = sampleNum - sampleNum_valid
        
        sampleNames_train_valid = self.shuffleSampleNames_all(sampleNames_train_valid)
        
        return sampleNames_train_valid[0:self.sampleNum_train], sampleNames_train_valid[self.sampleNum_train:sampleNum]            
        #return sampleNames_train, sampleNames_valid
        
        

    def getN_folds(self):
        self.getN_folds()
        
    def getCVindex(self):
        return self.cvIndex