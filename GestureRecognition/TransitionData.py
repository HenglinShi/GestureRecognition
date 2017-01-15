'''
Created on Dec 16, 2016

@author: hshi
'''
from Data import Data
from DataPreparation import estimatePrior
import os
class TransitionData(Data):
    '''
    classdocs
    '''


    def __init__(self,
                 dataDir,
                 dataName,
                 classTypeNum,
                 stateTypeNumPerClass):
        '''
        Constructor
        '''
        
        super(TransitionData, self).__init__(dataDir, dataName)
        
        self.classTypeNum = classTypeNum
        self.stateTypeNumPerClass = stateTypeNumPerClass
        self.transitionMatrix = None
        self.prior = None
        self.path = None
        
        self.transitionMatrix, self.prior = self.estimateTransitionData()
        self.save()
        
    def estimateTransitionData(self):
        return estimatePrior(self.dir, self.classTypeNum, self.stateTypeNumPerClass)
    
    def save(self, path = None):
        if path == None:
            self.path = os.path.join(self.dir, os.path.pardir, self.name + '.pkl')
            
    def getPath(self):
        return self.path
    
    def getPrior(self):
        return self.prior
    
    def getTransitionMatrix(self):
        return self.transitionMatrix