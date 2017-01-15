'''
Created on Dec 16, 2016

@author: hshi
'''
import caffe
import numpy as np
from utils import viterbi_path_log, Extract_feature_UNnormalized, Extract_feature_Realtime
from ChalearnLAPSample import GestureSample
import os

from sklearn import preprocessing

def testOneNet(netPath, 
               weightPath, 
               transitionData, 
               scalar, 
               dataDir,
               stateTypeNumAll, 
               stateTypeNumPerClass, 
               batchSize,
               sampleWisePreNormalization,
               subjectWisePreNormalization,
               transitionEstimationType,
               normalizationType,
               actionUnits):
    
    net = caffe.Net(netPath, weightPath, caffe.TEST)
    
    samples = os.listdir(dataDir)
    sampleNum = len(samples)
    
    groundTruths = np.zeros(sampleNum * 100, dtype = np.int32)
    predictions = np.zeros(sampleNum * 100, dtype = np.int32)
    
    paths = list()
    sampleNames_test = list()
    
    globalIte = 0
        
    for gestureIte, sample in enumerate(samples):         
        print("\t Processing file " + sample)
        
        smp=GestureSample(os.path.join(dataDir,sample))
        
        
        gesturesList=smp.getGestures()
        for gesture in gesturesList:
            
            
            
            gestureID,startFrame,endFrame=gesture
            Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, actionUnits, startFrame, endFrame)           

            if not valid_skel:
                print "No detected Skeleton: ", gestureID
            else:    
                                        
                Feature = Extract_feature_Realtime(Skeleton_matrix, len(actionUnits))
                
                if sampleWisePreNormalization:
                    scaler = preprocessing.StandardScaler().fit(Feature)
                    Feature = scaler.transform(Feature)
                 
                if subjectWisePreNormalization:
                    pass
                else:
                    pass            
                            
                            
                            
                if normalizationType != 0:  
                    Feature = scalar.transform(Feature)
                             
                emssionMatrixFinal = forwardOneSample(net, batchSize, Feature, stateTypeNumAll)        
                print("\t Viterbi path decoding " )
           
                [currentPath, _, _] = viterbi_path_log(np.log(transitionData.getPrior()), 
                                                       np.log(transitionData.getTransitionMatrix()), 
                                                       emssionMatrixFinal)
                currentPrediction = vertibDecoding(currentPath, stateTypeNumPerClass)
        
        
                predictions[globalIte] = currentPrediction
                groundTruths[globalIte] = gestureID
                paths.append(currentPath)
                
                sampleNames_test.append(sample)
                globalIte += 1
                    
                print(currentPrediction)
                print(gestureID) 
        del smp        

    return predictions[0:globalIte], groundTruths[0:globalIte], paths, sampleNames_test
 
        
        
    

def getNextSample(sample, label, clipMarker):
    
    if sample.shape[0] > 0:
        if clipMarker[0] == 0:
            tmp = clipMarker[0:len(clipMarker)]
        
            for i in range(len(tmp)):
                if tmp[i] == 0:
                    break
            
            return i
        
        return 0
    
    return 0
        
    
def forwardOneSample(net, batchSize, Feature, stateTypeNumAll):
        
    if Feature.shape[0] % batchSize != 0:
        input_feature = np.zeros(shape = (batchSize * (Feature.shape[0]/batchSize + 1),Feature.shape[1]), dtype=np.float)
        input_feature[0:Feature.shape[0],:] = Feature
                                
    else:
        input_feature = Feature
                                
    input_cm = np.ones(input_feature.shape[0], dtype=np.uint8)
    input_cm[0] = 0                
    emssionMatrix = np.zeros(shape=(stateTypeNumAll, input_feature.shape[0]), dtype=np.float)
                               
    # GET LSTM OUTPUT 
    for i in range(input_feature.shape[0]/batchSize):

        net.blobs['sample'].data[...] = input_feature[batchSize * i:batchSize * (i + 1),:].reshape([batchSize, 1, 1, -1])    
        net.blobs['clip_marker'].data[...] = input_cm [batchSize * i:batchSize * (i + 1)].reshape([batchSize, 1])
        net.forward(start='sample_1')             
                    
        emssionMatrix[:,batchSize * i:batchSize * (i + 1)] = net.blobs['lstm_1'].data.T
                                              
    return emssionMatrix[:,0:Feature.shape[0]]


def vertibDecoding(currentPath, stateTypeNum_perClass):
    
    begFrameState = currentPath[0]      
                     
    if begFrameState % stateTypeNum_perClass == 0:
                         
        if 1: #endFrameState == begFrameState + statusTypeNumPerClass - 1:
                                         
            currentPrediction = begFrameState / stateTypeNum_perClass + 1
                             
            for i in range(len(currentPath) - 1):
                if (currentPath[i+1] - currentPath[i]) != 1 and 0:
                    currentPrediction = -1
                    break;
                                     
        else:
            currentPrediction = -1
                                      
    else:
        currentPrediction = -1
        
    return currentPrediction
