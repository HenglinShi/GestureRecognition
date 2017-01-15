'''
Created on Dec 16, 2016

@author: hshi
'''
from Data import Data

class ThreeDimensionExpressionActionUnitData(Data):
    '''
    classdocs
    '''


    def __init__(self,
                 dataDir,
                 dataName,
                 actionUnits,
                 stateTypeNumPerClass,
                 classTypeNum):
        '''
        Constructor
        '''
        super().__init__(dataDir, dataName)
        
        self.actionUnits = actionUnits
        self.stateTypeNumPerClass
        self.classTypeNum = classTypeNum
        self.stateTypeNumAll = self.stateTypeNumPerClass * self.classTypeNum
        
        self.sample, self.label = self.extractFeature()
        
        
    
        
    def extractFeature(self):
        
        
        return 0
        
        
    def getSample(self):
        return self.sample
    
    def getLabel(self):
        return self.label
    
    def getSampleNum(self):
        return self.sample