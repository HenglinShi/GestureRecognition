'''
Created on Dec 16, 2016

@author: hshi
'''

class Tuples(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.validAccuracy = None
        self.weightPath = None
        self.sampleNames_train = None
        self.sampleNames_valid = None
        self.sampleNames_test = None
        self.workingDir = None
        self.transitionEstimationType = -1
        self.subjectWisePreNormalization = -1
        self.sampleWisePreNormalization = -1
        self.normalizationType = -1
        self.batchSize = -1
        self.trainingEpoches = -1
        self.stateTypeNumPerClass = -1 
        self.bestWeightInd =  -1
        self.bestValidAccuracy = -1.0
        self.bestWeightPath = None
        self.HMMpaths = None
        self.testAccuracy = None
        self.bestWeightTestAccuracy = -1.0
        self.ratio_train = 0
        self.ratio_valid = 0
        self.ratio_test = 0
        
        self.foldsNum = 0
        self.label_test = 0
        self.validLoss = 0.0
        
        
    def setValidLoss(self, validLoss):
        self.validLoss = validLoss
        
    def getValidLoss(self):
        return self.validLoss

        
    def setStateNumPerClass(self, stateNumPerClass):
        self.stateTypeNumPerClass = stateNumPerClass
        
    def getStateNumPerClass(self):
        return self.getStateNumPerClass()
        
    def setLabel_test(self, label_test):
        self.label_test = label_test
        
    def getLabel_test(self):
        return self.label_test
        
        
    def setRatio_train(self, ratio_train):
        self.ratio_train = ratio_train
        
    def getRatio_train(self):
        return self.ratio_train
    
    def setRatio_valid(self, ratio_valid):
        self.ratio_valid = ratio_valid
        
    def getRatio_valid(self):
        return self.ratio_valid
    
    def setRatio_test(self, ratio_test):
        self.ratio_test = ratio_test
        
    def getRatio_test(self):
        return self.ratio_test
    
    def setFoldsNum(self, foldsNum):
        self.foldsNum = foldsNum
        
    def getFoldsNum(self):
        return self.foldsNum
        
    def setBestWeightTestAccuracy(self, bestWeightTestAccuracy):
        self.bestWeightTestAccuracy = bestWeightTestAccuracy
        
    def getBestWeightTestAccuracy(self):
        return self.bestWeightTestAccuracy
        
    def setTestAccuracy(self, testAccuracy):
        self.testAccuracy = testAccuracy
        
    def getTestAccuracy(self):
        return self.testAccuracy
        
    def setHMMpaths(self, HMMpaths):
        self.HMMpaths = HMMpaths
        
    def getHMMpaths(self):
        return self.HMMpaths
        
    def setBestWeightPath(self, bestWeightPath):
        self.bestWeightPath = bestWeightPath
        
    def getBestWeightPath(self):
        return self.bestWeightPath
    
    def setBestValidAccuracy(self, bestValidAccuracy):
        self.bestValidAccuracy = bestValidAccuracy
        
    def getBestValidAccuracy(self):
        return self.bestValidAccuracy
        
        
    def setBestWeightInd(self, bestWeightInd):
        self.bestWeightInd = bestWeightInd
        
    def getBestWeightInd(self):
        return self.bestWeightInd
        
        
    def setStateTypeNumPerClass(self, stateTypeNumPerClass):
        self.stateTypeNumPerClass = stateTypeNumPerClass
        
    def getStateTypeNumPerClass(self):
        return self.stateTypeNumPerClass
        
    def setTrainingEpoches(self, trainingEpoches):
        self.trainingEpoches = trainingEpoches
        
    def getTrainingEpoches(self):
        return self.trainingEpoches
    
    
    def setBatchSize(self, batchSize):
        self.batchSize = batchSize
        
    def getBatchSize(self):
        return self.batchSize
    
        
    def setValidAccuracy(self, validAccuracy):
        self.validAccuracy = validAccuracy
        
    def getValidAccuracy(self):
        return self.validAccuracy
    
    
    def setWeightPath(self, weightPath):
        self.weightPath = weightPath
        
    def getWeightPath(self):
        return self.weightPath
    
    def setSampleNames_train(self, sampleNames_train):
        self.sampleNames_train = sampleNames_train
        
    def getSampleNames_train(self):
        return self.sampleNames_train
    
    def setSampleNames_valid(self, sampleNames_valid):
        self.sampleNames_train = sampleNames_valid
        
    def getSampleNames_valid(self):
        return self.sampleNames_valid
    
    def setSampleNames_test(self, sampleNames_test):
        self.sampleNames_test = sampleNames_test
        
    def getSampleNames_test(self):
        return self.sampleNames_test
    
    def setWorkingDir(self, workingDir):
        self.workingDir = workingDir
        
    def getWorkingDir(self):
        return self.workingDir
    
    def setTransitionEstimationType(self, transitionEstimationType):
        self.transitionEstimationType = transitionEstimationType
        
    def getTransitionEstimationType(self):
        return self.transitionEstimationType
    
    
    def setSampleWisePreNormalization(self, sampleWisePreNormalization):
        self.sampleWisePreNormalization = sampleWisePreNormalization
        
    def getSampleWisePreNormalization(self):
        return self.sampleWisePreNormalization
    
    def setSubjectWisePreNormalization(self, subjectWisePreNormalization):
        self.subjectWisePreNormalization = subjectWisePreNormalization
        
    def getSubjectWisePreNormalization(self):
        return self.subjectWisePreNormalization
    
    def setNormalizationType(self, normalizationType):
        self.normalizationType = normalizationType
        
    def getNormalizationType(self):
        return self.normalizationType
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    