'''
Created on Dec 16, 2016

@author: hshi
'''
from DataPreparation import extractFeature
from Data import Data
import cPickle as pickle
import os
class ThreeDimensionExpressionActionUnitLstmData(Data):
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
        self.stateTypeNumPerClass = stateTypeNumPerClass
        self.classTypeNum = classTypeNum
        
        self.stateTypeNumAll = self.stateTypeNumPerClass * self.classTypeNum
        self.sample = None
        self.label = None
        self.clipMarker = None
        self.sample, self.label, self.clipMarker = self.extractFeature()
        
        self.path = None
        
        self.save()
        
    def extractFeature(self):
        return extractFeature(self.dir, self.actionUnits, self.classTypeNum, self.stateTypeNumPerClass)
        
        
    def getSample(self):
        return self.sample
    
    def getLabel(self):
        return self.label
    
    def getSampleNum(self):
        return self.sample
    
    def save(self, path = None):
        if path == None:
            self.path = os.path.join(self.dir, os.path.pardir, self.name + '.pkl')
            
        else:
            self.path = path
            
        f = open(self.path,'wb')
            
        pickle.dump( {"sample": self.sample, 
                      "label": self.label,
                      "clipMarker": self.clipMarker},f)
        f.close() 
        
        
    def getPath(self):
        return self.path