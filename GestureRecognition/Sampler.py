'''
Created on Dec 15, 2016

@author: hshi
'''

class Sampler(object):
    '''
    classdocs
    '''


    def __init__(self,
                 sampleNames_all = None,
                 ratio_train = 0.8,
                 ratio_valid = 0.2):
        '''
        Constructor
        '''
        self.sampleNames_all = sampleNames_all
        self.ratio_train = ratio_train
        self.ratio_valid = ratio_valid
        self.sampleNum = 0
        self.sampleNum_train = 0
        self.sampleNum_valid = 0
        self.sampleNum_test = 0
        
    def getSampleNames_all(self):
        return self.sampleNames_all