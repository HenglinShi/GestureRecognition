'''
Created on Dec 15, 2016

@author: hshi
'''
from SamplingResult import SamplingResult

class CrossValidatingRandomSamplingResult(SamplingResult):
    '''
    classdocs
    '''


    def __init__(self,
                 sampleNames_train = None,
                 sampleNames_valid = None,
                 sampleNames_test = None,
                 foldIte = 1):
        '''
        Constructor
        '''
        super(CrossValidatingRandomSamplingResult, self).__init__(sampleNames_train, sampleNames_valid, sampleNames_test)
        
        self.foldIte = foldIte
        
    def getCVite(self):
        return self.foldIte