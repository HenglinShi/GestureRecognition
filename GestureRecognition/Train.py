'''
Created on Dec 16, 2016

@author: hshi
'''
import os
import caffe
import numpy as np
from DataPreparation import myMkdir, moveFile, extractFeature
from DefineNet import defineNet
from DefineSolver import defineSolver
from sklearn import preprocessing
import cPickle as pickle
from CrossValidatingRandomSampler import CrossValidatingRandomSampler
from ThreeDimensionExpressionActionUnitLstmData import ThreeDimensionExpressionActionUnitLstmData
from TransitionData import TransitionData

def train_earlyStop(weightDir, solverPath, netPath_train, netPath_valid, batchSize, frameNum_train, frameNum_valid, trainingEpoches, data_valid):
    sample_valid = data_valid['sample']
    label_valid = data_valid['label']
    label_valid = label_valid[:,0]
    clipMarker_valid = data_valid['clipMarker']
    
    
    batchNumPerEpoch_train = frameNum_train / batchSize + 1
 
    
    solver = None
    solver = caffe.SGDSolver(solverPath)
    
    
    iterations_train = trainingEpoches * batchNumPerEpoch_train
    #testInterval = batchNumPerEpoch_train
    validAccuracy = np.zeros(trainingEpoches, dtype = np.float)
    validLoss = np.zeros(trainingEpoches, dtype = np.float)
    weightFilePath = list()


    trainedEpoches = 0

    bestValidLoss = np.inf
    improvementThreshold = 0.999
    
    patience = 4
    patience_increase = 2.
    
    
    while (trainedEpoches < trainingEpoches) and (patience >= trainedEpoches):
                
        solver.step(batchNumPerEpoch_train)
        currentValidAccuracy, currentValidLoss = validOneEpoche(solver.test_nets[0],
                                                      sample_valid,
                                                      label_valid,
                                                      clipMarker_valid,
                                                      batchSize)   
        
        
        if currentValidLoss < bestValidLoss:
            if currentValidLoss< bestValidLoss * improvementThreshold:
                patience = max(patience, patience * patience_increase)
            
            bestValidLoss = currentValidLoss
        
        validAccuracy[trainedEpoches] = currentValidAccuracy
        validLoss[trainedEpoches] = currentValidLoss
            
            
        currentWeightFilePath = os.path.join(weightDir, str(currentValidAccuracy) + "_" + str(trainedEpoches) + ".caffemodel")      
        weightFilePath.append(currentWeightFilePath)     
        solver.net.save(currentWeightFilePath)
        print 'test after training', trainedEpoches, 'epoches...'   
        print(currentValidAccuracy)
        print(currentValidLoss)
        
        trainedEpoches += 1

    return validAccuracy[0:trainedEpoches], weightFilePath, validLoss[0:trainedEpoches]


def train(weightDir, solverPath, netPath_train, netPath_valid, batchSize, frameNum_train, frameNum_valid, trainingEpoches, data_valid):
    
    sample_valid = data_valid['sample']
    label_valid = data_valid['label']
    label_valid = label_valid[:,0]
    clipMarker_valid = data_valid['clipMarker']
    
    
    batchNumPerEpoch_train = frameNum_train / batchSize + 1
 
    
    solver = None
    solver = caffe.SGDSolver(solverPath)
    
    
    iterations_train = trainingEpoches * batchNumPerEpoch_train
    testInterval = batchNumPerEpoch_train
        
    validAccuracy = np.zeros(int(np.ceil(iterations_train / batchNumPerEpoch_train)))
    validLoss = np.zeros(int(np.ceil(iterations_train / batchNumPerEpoch_train)))
    
    weightFilePath = list()
    
    for iter_train in range(iterations_train):
            
        solver.step(1)
            
        if iter_train % testInterval == 0:
            print 'test after training', iter_train / testInterval, 'epoches...'   
            currentValidAccuracy, currentValidLoss = validOneEpoche(solver.test_nets[0],
                                                      sample_valid,
                                                      label_valid,
                                                      clipMarker_valid,
                                                      batchSize)   
                    

               
               
            print(currentValidAccuracy)  
             
            validAccuracy[iter_train // testInterval] = currentValidAccuracy
            validLoss[iter_train // testInterval] = currentValidLoss
            
            
            currentWeightFilePath = os.path.join(weightDir, str(currentValidAccuracy) + "_" + str(iter_train) + ".caffemodel")      
            weightFilePath.append(currentWeightFilePath)     
            solver.net.save(currentWeightFilePath)

    return validAccuracy, weightFilePath, validLoss




def validOneEpoche(net, sample, label, clipMarker, batchSize):
        
    sampleNum = sample.shape[0]
        
    inputSample = np.zeros(shape = (batchSize * np.ceil(sampleNum * 1.0 /batchSize), sample.shape[1]), dtype = np.float)
        
    inputSample[0:sampleNum] = sample
        
    inputCM = np.zeros(inputSample.shape[0], dtype = np.uint8)
        
    inputCM[0:sampleNum] = clipMarker[:,0]
        
    predictions = np.zeros(inputSample.shape[0], dtype = np.uint8)
       
    loss = 0.0
       
        
    for i in range(inputSample.shape[0]/batchSize):
        net.blobs['sample'].data[...] = inputSample[batchSize * i:batchSize * (i + 1),:].reshape([batchSize, 1, 1, -1])    
        net.blobs['clip_marker'].data[...] = inputCM [batchSize * i:batchSize * (i + 1)].reshape([batchSize, 1])
        net.forward(start='sample_1') 
                              
        predictions[batchSize * i:batchSize * (i + 1)] = net.blobs['lstm_1'].data.argmax(1)
        loss = loss + net.blobs['loss'].data

    predictions = predictions[0:sampleNum]
        
    correct = sum(predictions == label)
        
    return (correct * 1.0/sampleNum), loss/(inputSample.shape[0]/batchSize)
