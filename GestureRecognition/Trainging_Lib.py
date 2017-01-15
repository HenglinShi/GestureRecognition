'''
Created on Dec 6, 2016

@author: hshi
'''
'''
Created on Dec 5, 2016

@author: hshi
'''
import openpyxl as pyxl
import numpy as np
import caffe
import os
import scipy.io as sio
import cPickle as pickle
from sklearn import preprocessing
from DataPreparation import extractFeature, moveFile, estimatePrior,\
    loadPrior, loadTransitionMatrix, myMkdir, mySampling_subjectDependentValidating_subjectDependentTesting, \
    mySampling_subjectIndependentValidating_subjectIndependentTesting, mySampling_subjectDependentValidating_subjectIndependentTesting, saveResult, summarizingResult
from DefineNet import defineNet
from DefineSolver import defineSolver
from ChalearnLAPSample import GestureSample
from utils import Extract_feature_UNnormalized, viterbi_path_log, Extract_feature_Realtime, normalize


    
actionUnits = ['1', '2', '3', '4', '5', '6', '7', '8' ,'9', '10',
                   '11','12','13','14','15','16','17','18','19','20',
                   '21','22','23','24','25','26','27','28','29','30',
                   '31','32','33','34','35','36','37','38','39','40',
                   '41','42','43','44','45','46','47','48','49','50']  
     
classTypeNum = 3
stateTypeNum_perClass = 5    
stateTypeNum_all = classTypeNum * stateTypeNum_perClass    



MATLAB_FILE_POSTFIX = ".mat"
PYTHON_FILE_POSTFIX = ".pkl"
DATA_FILE_POSTFIX_ZIP = '.zip'
PROTOTXT_FILE_POSTFIX = ".prototxt"  
NET_NAME_TRAIN = "net_train"
NET_NAME_VALID = "net_valid"
NET_NAME_TEST = "net_test"
SOLVER_NAME = "solver"
    
DATA_FILE_NAME_TRAIN = "train"
DATA_FILE_NAME_VALID = "valid"
DATA_FILE_NAME_TEST = "test"
DB_NAME_SAMPLE = 'sample'
DB_NAME_LABEL = 'label'
DB_NAME_CLIP_MARKER = 'clipMarker'  

PARAMETER_FILE_NAME_PRIOR = "PRIOR"
PARAMETER_FILE_NAME_PRIOR_MAT = PARAMETER_FILE_NAME_PRIOR + MATLAB_FILE_POSTFIX
PARAMETER_FILE_NAME_PRIOR_PY = PARAMETER_FILE_NAME_PRIOR + PYTHON_FILE_POSTFIX
    
PARAMETER_FILE_NAME_TRANSITION_MATRIX= "TRANSITION_MATRIX"
PARAMETER_FILE_NAME_TRANSITION_MATRIX_MAT = PARAMETER_FILE_NAME_TRANSITION_MATRIX + MATLAB_FILE_POSTFIX
PARAMETER_FILE_NAME_TRANSITION_MATRIX_PY = PARAMETER_FILE_NAME_TRANSITION_MATRIX + PYTHON_FILE_POSTFIX
    
PARAMETER_FILE_NAME_MEAN_STD = "distribution_parameter"
PARAMETER_FILE_NAME_MEAN_STD_MAT = PARAMETER_FILE_NAME_MEAN_STD + MATLAB_FILE_POSTFIX
PARAMETER_FILE_NAME_MEAN_STD_PY = PARAMETER_FILE_NAME_MEAN_STD + PYTHON_FILE_POSTFIX
    
    
SUBJECT_INFO_FILE_NAME = "Label4SMIC3D"

def main():
    
    workingDir ="/research/tklab/personal/hshi/Gesture Recognition/Experiments/(Micro_Expression)_(LSTM+HMM)_20161207/"   
    dataDir = "/research/tklab/personal/hshi/Gesture Recognition/SMIC3D/" 
    ratios = np.zeros(3, dtype = np.float)
    ratios[0] = 0.64
    ratios[1] = 0.18
    ratios[2] = 0.18
    batchSize = 100
    #isBatchNormalization = 0
    trainingEpochs = 300
    preNormalize = 2
    reSample = 1
    isSubjectIndependentTesting = 1
    isSubjectIndependentValidation = 1
    if reSample:
        myMkdir(workingDir)
    
    #
    
    if isSubjectIndependentTesting:
        
        if isSubjectIndependentValidation:
            result = train_subjectIndependentValidating_subjectIndependentTesting(workingDir, dataDir, reSample, preNormalize, ratios, actionUnits, batchSize, trainingEpochs)
        else:
            result = train_subjectDependentValidating_subjectIndependentTesting(workingDir, dataDir, reSample, preNormalize, ratios, actionUnits, batchSize, trainingEpochs)
        
    else:
        result = train_subjectDependentValidating_subjectDependentTesting(workingDir, dataDir, reSample, preNormalize, ratios, actionUnits, batchSize, trainingEpochs)
        
        
    
    result = myTest_bestValid(workingDir, dataDir, result, preNormalize)
    
    # Write in files
    
    resultFilePath = os.path.join(workingDir, 'result.xlsx')
    resultFile = pyxl.Workbook()
    sheet1 = resultFile.active
    sheet1.title = 'sum'
 
    for i in range(len(result)):   
        currentResult = result[i]
        newsheet = resultFile.create_sheet(str(i + 1), i + 2) 
        newsheet = saveResult(newsheet, currentResult, isSubjectIndependentTesting, isSubjectIndependentValidation)
    
    sheet1 = summarizingResult(sheet1, result, isSubjectIndependentTesting, isSubjectIndependentValidation)

            
    resultFile.save(resultFilePath)
    
    
    print (12)
    
    #subjectDepedentTrainging

def myTest_bestValid(workingDir, dataDir_raw, result, preNormalization):
    dataDir_raw = os.path.join(dataDir_raw, 'data')
    
    if len(result) == 1:
        
        
        sampleName_test = result[0]['Testing Sample Names']
        weightFiles = result[0]['Weight Path']
        validAccuracy = result[0]['Validating Accuracy'] 
        batchSize = result[0]['Batch Size']
        bestWeightInd = validAccuracy.argmax()
        bestWeightPath = weightFiles[bestWeightInd]
        
        dataDir_test = os.path.join(workingDir, 'data', 'test')
        
        myMkdir(dataDir_test)
        moveFile(sampleName_test, dataDir_raw, dataDir_test)
        
        
        
        sample_test, label_test, clipMarker_test = extractFeature(dataDir_test, actionUnits, classTypeNum, stateTypeNum_perClass)
        dataFilePath_test = os.path.join(workingDir, 'data_test.pkl')  
        netPath_test = os.path.join(workingDir, NET_NAME_TEST + PROTOTXT_FILE_POSTFIX)

        f = open(dataFilePath_test,'wb')
        pickle.dump( {"sample": sample_test, 
                      "label": label_test,
                      "clipMarker": clipMarker_test},f)
        f.close() 
        
        dataFileParam_test = dict(data_path = dataFilePath_test, batch_size = batchSize)
        net_test = defineNet('test', batchSize, dataFileParam_test, stateTypeNum_all, 11175)
    
        with open(netPath_test, 'w') as f:
            f.write(str(net_test.to_proto()))
        
        net_test = caffe.Net(netPath_test, bestWeightPath, caffe.TEST)
        
        predictions, groundTruths, resultedPaths = test(workingDir, dataDir_test, net_test, batchSize, preNormalization)
        
        result[0]['Testing Accuracy'][bestWeightInd] = (float) (sum(predictions == groundTruths)) / len(groundTruths)
        
        
    else:
        for i in range(len(result)):
            
            currentWorkingDir = os.path.join(workingDir, str(i)) 
            batchSize = result[i]['Batch Size']
            weightFiles = result[i]['Weight Path']
            validAccuracy = result[i]['Validating Accuracy']
            
            bestWeightInd = validAccuracy.argmax()
            bestWeightPath = weightFiles[bestWeightInd]
            
            dataDir_test = os.path.join(currentWorkingDir, 'data', 'test')
            sampleName_test = result[i]['Testing Sample Names']  
            myMkdir(dataDir_test)
            moveFile(sampleName_test, dataDir_raw, dataDir_test)
            sample_test, label_test, clipMarker_test = extractFeature(dataDir_test, actionUnits, classTypeNum, stateTypeNum_perClass)
            
            
            
            netPath_test = os.path.join(currentWorkingDir, NET_NAME_TEST + PROTOTXT_FILE_POSTFIX)
            dataFilePath_test = os.path.join(currentWorkingDir, 'data_test.pkl')  
   
            
            f = open(dataFilePath_test,'wb')
            pickle.dump( {"sample": sample_test, 
                      "label": label_test,
                      "clipMarker": clipMarker_test},f)
            f.close() 
        
            dataFileParam_test = dict(data_path = dataFilePath_test, batch_size = batchSize)
            net_test = defineNet('test', batchSize, dataFileParam_test, stateTypeNum_all, 11175)
    
            with open(netPath_test, 'w') as f:
                f.write(str(net_test.to_proto()))

            net_test = caffe.Net(netPath_test, bestWeightPath, caffe.TEST)  
            
            predictions, groundTruths, resultedPaths = test(currentWorkingDir, dataDir_test, net_test, batchSize, preNormalization)
            
            result[i]['Testing Accuracy'][bestWeightInd] = (float) (sum(predictions == groundTruths)) / len(groundTruths)
              
    return result      

def test(workingDir, dataDir_test, net_test, batchSize, preNormalization):
    
  
    prior = loadPrior(os.path.join(workingDir, PARAMETER_FILE_NAME_PRIOR_MAT))
    transitionMatrix = loadTransitionMatrix(os.path.join(workingDir, PARAMETER_FILE_NAME_TRANSITION_MATRIX_MAT))
    
    if preNormalization != 0: 
        f = open(os.path.join(workingDir, PARAMETER_FILE_NAME_MEAN_STD + PYTHON_FILE_POSTFIX))
        SK_normalization = pickle.load(f)
        mean = SK_normalization ['Mean']
        std = SK_normalization['Std']  
        f.close()
        
     
       
    samples=os.listdir(dataDir_test)
    sampleNum = len(samples)
    
    groundTruths = np.zeros(sampleNum, dtype = np.int32)
    predictions = np.zeros(sampleNum, dtype = np.int32)

    
    resultedPaths = list()

    gestureIte = 0

    for file_count, sample in enumerate(samples):         
        print("\t Processing file " + sample)
        
        smp=GestureSample(os.path.join(dataDir_test,sample))

        gesturesList=smp.getGestures()
        for gesture in gesturesList:
                    
            gestureID,startFrame,endFrame=gesture
            Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, actionUnits, startFrame, endFrame)           

            if not valid_skel:
                print "No detected Skeleton: ", gestureID
            else:                            
                Feature = Extract_feature_Realtime(Skeleton_matrix, len(actionUnits))
                            
                if preNormalization < 2:  
                    Feature = normalize(Feature, mean, std)
                             
                emssionMatrixFinal = forwardOneSample(net_test, batchSize, Feature)        
                print("\t Viterbi path decoding " )
           
                [currentPath, predecessor_state_index, global_score] = viterbi_path_log(np.log(prior), np.log(transitionMatrix), emssionMatrixFinal)
                currentPrediction = vertibDecoding(currentPath, stateTypeNum_perClass)
        
                predictions[gestureIte] = currentPrediction
                groundTruths[gestureIte] = gestureID
                
                gestureIte = gestureIte + 1


                resultedPaths.append(currentPath)
                    
                print(currentPrediction)
                print(gestureID) 
        del smp        

    return predictions[0:gestureIte], groundTruths[0:gestureIte], resultedPaths
 
    
def forwardOneSample(net, batchSize, Feature):
        
    if Feature.shape[0] % batchSize != 0:
        input_feature = np.zeros(shape = (batchSize * (Feature.shape[0]/batchSize + 1),Feature.shape[1]), dtype=np.float)
        input_feature[0:Feature.shape[0],:] = Feature
                                
    else:
        input_feature = Feature
                                
    input_cm = np.ones(input_feature.shape[0], dtype=np.uint8)
    input_cm[0] = 0                
    emssionMatrix = np.zeros(shape=(stateTypeNum_all, input_feature.shape[0]), dtype=np.float)
                               
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

def train_subjectDependentValidating_subjectIndependentTesting(workingDir, dataDir, reSample, preNormalize, ratios, actionUnits, batchSize, trainingEpochs):
    
    result = list()
    
    if reSample:
        sampleDir = os.path.join(dataDir, 'data')
        
        SUBJECT_INFO_FILE_PATH = os.path.join(dataDir, SUBJECT_INFO_FILE_NAME)
        sampleSubject = sio.loadmat(SUBJECT_INFO_FILE_PATH)['sub_label']   
        subjectName = np.unique(sampleSubject)
        subjectNum = len(subjectName)
        
        for subjectIte in range(subjectNum):
            
            sampleName_train, \
            sampleName_valid, \
            sampleName_test, \
            subjectName_train_valid, \
            subjectName_test = mySampling_subjectDependentValidating_subjectIndependentTesting(sampleDir, 
                                                                                               sampleSubject, 
                                                                                               subjectIte, 
                                                                                               ratios, 
                                                                                               DATA_FILE_POSTFIX_ZIP)
            
            currentWorkingDir = os.path.join(workingDir, str(subjectIte))
            myMkdir(currentWorkingDir)      
            validAccuracy, weightFilePath, netPath_valid = train(sampleDir, currentWorkingDir, sampleName_train, sampleName_valid, preNormalize, batchSize, trainingEpochs)  
        
        
            
            currentResult = {'Validating Accuracy': validAccuracy, 
                             'Weight Path': weightFilePath,
                             'Testing Accuracy': np.zeros(shape = validAccuracy.shape, dtype = np.float),
                             'Batch Size': batchSize,
                             'Training Epochs': trainingEpochs,
                             'PreNormalization': preNormalize,
                             'Batch Normalization': 'yes',
                             'Training and Subject Subject Names': subjectName_train_valid,
                             'Testing Subject Names': subjectName_test,
                             'Training Sample Names': sampleName_train,
                             'Validating Sample Names': sampleName_valid,
                             'Testing Sample Names': sampleName_test,
                             'Training Ratio': ratios[0],
                             'Validating Ratio': ratios[1],
                             'Test Net Path': netPath_valid}
        

        
            result.append(currentResult)
            
    return result
            
            
            
def train_subjectIndependentValidating_subjectIndependentTesting(workingDir, dataDir, reSample, preNormalize, ratios, actionUnits, batchSize, trainingEpochs):
    
    result = list()
    
    if reSample:
        sampleDir = os.path.join(dataDir, 'data')
    
        SUBJECT_INFO_FILE_PATH = os.path.join(dataDir, SUBJECT_INFO_FILE_NAME)
        sampleSubject = sio.loadmat(SUBJECT_INFO_FILE_PATH)['sub_label']   
        subjectName = np.unique(sampleSubject)
        subjectNum = len(subjectName)

        for subjectIte in range(subjectNum):

            sampleName_train, \
            sampleName_valid, \
            sampleName_test, \
            subjectName_train, \
            subjectName_valid, \
            subjectName_test \
                = mySampling_subjectIndependentValidating_subjectIndependentTesting(sampleDir, 
                                                                                    sampleSubject, 
                                                                                    subjectIte, 
                                                                                    ratios, 
                                                                                    DATA_FILE_POSTFIX_ZIP)
            
            
            currentWorkingDir = os.path.join(workingDir, str(subjectIte))
            myMkdir(currentWorkingDir)
            validAccuracy, weightFilePath, netPath_valid = train(sampleDir, 
                                                   currentWorkingDir, 
                                                   sampleName_train, 
                                                   sampleName_valid, 
                                                   preNormalize, 
                                                   batchSize, 
                                                   trainingEpochs)  
                        
        
            
            currentResult = {'Validating Accuracy': validAccuracy, 
                             'Weight Path': weightFilePath,
                             'Testing Accuracy': np.zeros(shape = validAccuracy.shape, dtype = np.float),
                             'Batch Size': batchSize,
                             'Training Epochs': trainingEpochs,
                             'PreNormalization': preNormalize,
                             'Batch Normalization': 'yes',
                             'Training Subject Names': subjectName_train,
                             'Validating Subject Names': subjectName_valid,
                             'Testing Subject Names': subjectName_test,
                             'Training Sample Names': sampleName_train,
                             'Validating Sample Names': sampleName_valid,
                             'Testing Sample Names': sampleName_test,
                             'Training Ratio': ratios[0],
                             'Validating Ratio': ratios[1],
                             'Test Net Path': netPath_valid}
        
        
        
            result.append(currentResult)
        
    return result    



def train_subjectDependentValidating_subjectDependentTesting(workingDir, dataDir, reSample, preNormalize, ratios, actionUnits, batchSize, trainingEpochs):
    
    result = list()
    if reSample:
        
        dataDir = os.path.join(dataDir, 'data')
  
        sampleName_train, sampleName_valid, sampleName_test = mySampling_subjectDependentValidating_subjectDependentTesting(dataDir, ratios, DATA_FILE_POSTFIX_ZIP)
           
        f = open(workingDir + 'samplingInfo' + PYTHON_FILE_POSTFIX,'wb')
        pickle.dump( {"sampleName_train": sampleName_train, 
                      "sampleName_valid": sampleName_valid, 
                      "sampleName_test": sampleName_test},f)
        f.close()   
           
            
        validAccuracy, weightFilePath, netPath_valid = train(dataDir, workingDir, sampleName_train, sampleName_valid, preNormalize, batchSize, trainingEpochs)  
        
        currentResult = {'Validating Accuracy': validAccuracy, 
                         'Weight Path': weightFilePath,
                         'Testing Accuracy': np.zeros(shape = validAccuracy.shape, dtype = np.float),
                         'Batch Size': batchSize,
                         'Training Epochs': trainingEpochs,
                         'PreNormalization': preNormalize,
                         'Batch Normalization': 'yes',
                         'Training Sample Names': sampleName_train,
                         'Validating Sample Names': sampleName_valid,
                         'Testing Sample Names': sampleName_test,
                         'Training Ratio': ratios[0],
                         'Validating Ratio': ratios[1],
                         'Testing Ratio': ratios[2],
                         'Test Net Path': netPath_valid}
        
    
    else:
        validAccuracy, weightFilePath, netPath_valid = train_nonResample(workingDir, batchSize, trainingEpochs)      
        
        f = open(workingDir + 'samplingInfo' + PYTHON_FILE_POSTFIX)
        samplingInfo = pickle.load(f)
        sampleName_test = samplingInfo ['sampleName_test']
        f.close()

        
        currentResult = {'Validating Accuracy': validAccuracy, 
                         'Weight Path': weightFilePath,
                         'Testing Accuracy': np.zeros(shape = validAccuracy.shape, dtype = np.float),
                         'Batch Size': batchSize,
                         'Training Epochs': trainingEpochs,
                         'PreNormalization': preNormalize,
                         'Batch Normalization': 'yes',
                         'Training Sample Names': 0,
                         'Validating Sample Names': 0,
                         'Testing Sample Names': sampleName_test,
                         'Training Ratio': ratios[0],
                         'Validating Ratio': ratios[1],
                         'Testing Ratio': ratios[2],
                         'Test Net Path': netPath_valid}
        
        
        
    result.append(currentResult)
        
    
      
    return result


def prepareWorkingEnvironment(workingDir):
    
    weightDir = os.path.join(workingDir, 'weight')
    myMkdir(weightDir)
    
    dataDir = os.path.join(workingDir, 'data')
    myMkdir(dataDir)
    
    dataDir_train = os.path.join(dataDir, 'train')
    myMkdir(dataDir_train)
    
    dataDir_valid = os.path.join(dataDir, 'valid')
    myMkdir(dataDir_valid)
    
    return weightDir, dataDir_train, dataDir_valid

def myNormalization(sample_train, sample_valid, normalizationType):
    
    if normalizationType == 2:
        sample_train_valid = np.concatenate((sample_train, sample_valid), axis = 0)               
        scaler = preprocessing.StandardScaler().fit(sample_train_valid)
        mean = scaler.mean_
        std = scaler.scale_
          
        sample_train = scaler.transform(sample_train)
        sample_valid = scaler.transform(sample_valid)
        
    if normalizationType == 1:
                  
        scaler = preprocessing.StandardScaler().fit(sample_train)
        mean = scaler.mean_
        std = scaler.scale_
          
        sample_train = scaler.transform(sample_train)
        sample_valid = scaler.transform(sample_valid)
        
    if normalizationType == 0:
        mean = 0.0
        std = 0.0
        
    return sample_train, sample_valid, mean, std

def train_nonResample(workingDir, batchSize, trainingEpochs):

    weightDir = os.path.join(workingDir, 'weight')

    netPath_train = workingDir + NET_NAME_TRAIN + PROTOTXT_FILE_POSTFIX
    netPath_valid = workingDir + NET_NAME_VALID + PROTOTXT_FILE_POSTFIX
    solverPath = workingDir + SOLVER_NAME + PROTOTXT_FILE_POSTFIX
    
    dataFilePath_train = os.path.join(workingDir, 'data_train.pkl')  
    dataFilePath_valid = os.path.join(workingDir, 'data_valid.pkl')

  
    dataFileParam_train = dict(data_path = dataFilePath_train, batch_size = batchSize)
    dataFileParam_valid = dict(data_path = dataFilePath_valid, batch_size = batchSize)

    net_train = defineNet('train', batchSize, dataFileParam_train, stateTypeNum_all, 11175)
    net_valid = defineNet('test', batchSize, dataFileParam_valid, stateTypeNum_all, 11175)
    solver = defineSolver(netPath_train, netPath_valid, 0.01, './')
    
    with open(netPath_train, 'w') as f:
        f.write(str(net_train.to_proto()))
      
    with open(netPath_valid, 'w') as f:   
        f.write(str(net_valid.to_proto()))
    
    with open(solverPath, 'w') as f:
        f.write(str(solver))

        
    f1 = open(dataFilePath_train)
    data1 = pickle.load(f1)
      
    f2 = open(dataFilePath_valid)
    data2 = pickle.load(f2)
    
    frameNum_train = data1['sample'].shape[0]
    frameNum_valid = data2['sample'].shape[0]

    f1.close()
    f1.close()
    

    solver = None
    solver = caffe.SGDSolver(solverPath)
        
    
        
    batchNumPerEpoch_train = frameNum_train / batchSize + 1
    batchNumPerEpoch_valid = frameNum_valid / batchSize + 1
        
    iterations_train = trainingEpochs * batchNumPerEpoch_train
    iterations_test = 1 * batchNumPerEpoch_valid
        
    testInterval = batchNumPerEpoch_train
        
    validAccuracy = np.zeros(int(np.ceil(iterations_train / batchNumPerEpoch_train)))
        

    weightFilePath = list()
        
    for iter_train in range(iterations_train):
            
        solver.step(1)
            
        if iter_train % testInterval == 0:
            print 'test after training', iter_train / testInterval, 'epoches...'
                
            correct = 0
                
            for iter_test in range(iterations_test):
                    
                solver.test_nets[0].forward()
        
                correct += sum(solver.test_nets[0].blobs['lstm_1'].data.argmax(1) == solver.test_nets[0].blobs['label'].data.reshape(batchSize))
                
            validAccuracy[iter_train // testInterval] = (float) (correct) / (batchSize * iterations_test)   
                
            currentWeightFilePath = os.path.join(weightDir, str((float) (correct) / (batchSize * iterations_test)) + "_" + str(iter_train) + ".caffemodel")
                    
            weightFilePath.append(currentWeightFilePath)
                    
            solver.net.save(currentWeightFilePath)
                  
            print((float) (correct) / (batchSize * iterations_test))        
   
    
    
    return validAccuracy, weightFilePath, netPath_valid


def train(DATA_DIR_RAW, workingDir, sampleName_train, sampleName_valid, normalizationType, batchSize, trainingEpochs):  
        
    weightDir, \
    dataDir_train, \
    dataDir_valid = prepareWorkingEnvironment(workingDir)


    
    #dbPath_sample_train = os.path.join(dbDir_train, DB_NAME_SAMPLE)
    #dbPath_label_train = os.path.join(dbDir_train, DB_NAME_LABEL)
    #dbPath_clipMarker_train = os.path.join(dbDir_train, DB_NAME_CLIP_MARKER)
    #dbPath_sample_valid = os.path.join(dbDir_valid, DB_NAME_SAMPLE)
    #dbPath_label_valid = os.path.join(dbDir_valid, DB_NAME_LABEL)
    #dbPath_clipMarker_valid = os.path.join(dbDir_valid, DB_NAME_CLIP_MARKER)
    
    netPath_train = os.path.join(workingDir, NET_NAME_TRAIN + PROTOTXT_FILE_POSTFIX)
    netPath_valid = os.path.join(workingDir, NET_NAME_VALID + PROTOTXT_FILE_POSTFIX)  
    solverPath = os.path.join(workingDir, SOLVER_NAME + PROTOTXT_FILE_POSTFIX)
    
    dataFilePath_train = os.path.join(workingDir, 'data_train.pkl')  
    dataFilePath_valid = os.path.join(workingDir, 'data_valid.pkl')
    distributionFilePath = os.path.join(workingDir, PARAMETER_FILE_NAME_MEAN_STD + PYTHON_FILE_POSTFIX)
    priorFilePath = os.path.join(workingDir, PARAMETER_FILE_NAME_PRIOR + MATLAB_FILE_POSTFIX)
    transitionMatrixFilePath = os.path.join(workingDir, PARAMETER_FILE_NAME_TRANSITION_MATRIX + MATLAB_FILE_POSTFIX)
       

    moveFile(sampleName_train, DATA_DIR_RAW, dataDir_train)              
    moveFile(sampleName_valid, DATA_DIR_RAW, dataDir_valid)   
    sample_train, label_train, clipMarker_train = extractFeature(dataDir_train, actionUnits, classTypeNum, stateTypeNum_perClass)
    sample_valid, label_valid, clipMarker_valid = extractFeature(dataDir_valid, actionUnits, classTypeNum, stateTypeNum_perClass)
    
    
    if normalizationType != 0:
        sample_train, sample_valid, mean, std = myNormalization(sample_train, sample_valid, normalizationType)
        
        f = open(distributionFilePath,'wb')
        pickle.dump( {"Mean": mean, "Std": std },f)
        f.close() 
        
    f = open(dataFilePath_train,'wb')
    pickle.dump( {"sample": sample_train, 
                  "label": label_train,
                  "clipMarker": clipMarker_train},f)
    f.close() 
    
    f = open(dataFilePath_valid,'wb')
    pickle.dump( {"sample": sample_valid, 
                  "label": label_valid,
                  "clipMarker": clipMarker_valid},f)
    f.close() 
 
    
    
    prior, transitionMatrix = estimatePrior(dataDir_train, classTypeNum, stateTypeNum_perClass) 
    sio.savemat(priorFilePath, {'Prior': prior})
    sio.savemat(transitionMatrixFilePath, {'Transition_matrix':transitionMatrix})  
    

         
                

    
    dataFileParam_train = dict(data_path = dataFilePath_train, batch_size = batchSize)
    dataFileParam_valid = dict(data_path = dataFilePath_valid, batch_size = batchSize)

    net_train = defineNet('train', batchSize, dataFileParam_train, stateTypeNum_all, 11175)
    net_valid = defineNet('test', batchSize, dataFileParam_valid, stateTypeNum_all, 11175)
    solver = defineSolver(netPath_train, netPath_valid, 0.01, './')
    
    with open(netPath_train, 'w') as f:
        f.write(str(net_train.to_proto()))
      
    with open(netPath_valid, 'w') as f:   
        f.write(str(net_valid.to_proto()))
    
    with open(solverPath, 'w') as f:
        f.write(str(solver))
            
            
            
    # TRAINNING THE LSTM
    
    solver = None
    solver = caffe.SGDSolver(solverPath)
        
    frameNum_train = sample_train.shape[0]
    frameNum_valid = sample_valid.shape[0]
        
    batchNumPerEpoch_train = frameNum_train / batchSize + 1
    batchNumPerEpoch_valid = frameNum_valid / batchSize + 1
        
    iterations_train = trainingEpochs * batchNumPerEpoch_train
    iterations_test = 1 * batchNumPerEpoch_valid
        
    testInterval = batchNumPerEpoch_train
        
    validAccuracy = np.zeros(int(np.ceil(iterations_train / batchNumPerEpoch_train)))
        

    weightFilePath = list()
        
    for iter_train in range(iterations_train):
            
        solver.step(1)
            
        if iter_train % testInterval == 0:
            print 'test after training', iter_train / testInterval, 'epoches...'
                
            correct = 0
                
            for iter_test in range(iterations_test):
                    
                solver.test_nets[0].forward()
        
                correct += sum(solver.test_nets[0].blobs['lstm_1'].data.argmax(1) == solver.test_nets[0].blobs['label'].data.reshape(batchSize))
                
            validAccuracy[iter_train // testInterval] = (float) (correct) / (batchSize * iterations_test)   
                
            currentWeightFilePath = os.path.join(weightDir, str((float) (correct) / (batchSize * iterations_test)) + "_" + str(iter_train) + ".caffemodel")
                    
            weightFilePath.append(currentWeightFilePath)
                    
            solver.net.save(currentWeightFilePath)
                  
            print((float) (correct) / (batchSize * iterations_test))        
   
    
    
    return validAccuracy, weightFilePath, netPath_valid


if __name__ == '__main__':
    main()