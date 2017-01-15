'''
Created on Dec 15, 2016

@author: hshi
'''

class SamplingResult(object):
    '''
    classdocs
    '''


    def __init__(self,
                 sampleNames_train = None,
                 sampleNames_valid = None,
                 sampleNames_test = None):
        '''
        Constructor
        '''
        self.sampleNames_train = sampleNames_train
        self.sampleNames_valid = sampleNames_valid
        self.sampleNames_test = sampleNames_test
        
        
        
    def getSampleNames_train(self):
        return self.sampleNames_train
    
    def getSampleNames_valid(self):
        return self.sampleNames_valid
    
    def getSampleNames_test(self):
        return self.sampleNames_test