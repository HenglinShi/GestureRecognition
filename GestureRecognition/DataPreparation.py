'''
Created on Nov 27, 2016

@author: hshi
'''
from caffe.proto import caffe_pb2
import numpy as np
from numpy import newaxis, dtype
import caffe
import os
import lmdb
import random as rd
import shutil
from caffe import layers as L, params as P
import scipy.io as sio
import cPickle as pickle
from sklearn import preprocessing



def sampling(ratio_train = 0.64,
             ratio_valid = 0.18,
             ratio_test = 0.18,
             sampleNames = None
             ):
    
    sampleNum = len(sampleNames)
        
    sampleNum_valid = np.floor(sampleNum * ratio_valid).astype('int') 
    sampleNum_test = np.floor(sampleNum * ratio_test).astype('int') 
    
    sampleNum_train = sampleNum - sampleNum_valid - sampleNum_test
        
    sampleInd = np.linspace(0, sampleNum - 1, sampleNum).astype('int') 
    rd.shuffle(sampleInd)
        
    sampleInd_train = sampleInd[0:sampleNum_train]
    sampleInd_valid = sampleInd[sampleNum_train:sampleNum_train + sampleNum_valid]
    sampleInd_test = sampleInd[sampleNum_train + sampleNum_valid: sampleNum]
        
    sampleNames_train = list()
        
    for i in range(sampleNum_train):
        sampleNames_train.append(sampleNames[sampleInd_train[i]])
            
    sampleNames_valid = list()
        
    for i in range(sampleNum_valid):
        sampleNames_valid.append(sampleNames[sampleInd_valid[i]])
            
    sampleNames_test = list()
        
    for i in range(sampleNum_test):
        sampleNames_test.append(sampleNames[sampleInd_test[i]])
        
        
    return sampleNames_train, sampleNames_valid, sampleNames_test



# @ dataDir
# @ sampleNames
# @ usedJoints
# @ classTypeNum: 
# @ stateNumPerClass: 


def extractFeature(dataDir,
                   usedJoints,
                   classTypeNum, 
                   stateNumPerClass,
                   sampleWisePreNormalization):
    
    from utils import Extract_feature_UNnormalized
    from ChalearnLAPSample import GestureSample
    from utils import Extract_feature_Realtime


    sampleWisePreNormalization = sampleWisePreNormalization
    
    jointNum = len(usedJoints)
    
    
    count = 0
    # pre-allocating the memory
    samples_all = np.zeros(shape=(400000, (jointNum*(jointNum-1)/2 + jointNum**2)*3),dtype=np.float)
    
    #labels_logical_all = np.zeros(shape=(20000, stateNumPerClass*classTypeNum), dtype=np.uint8)
    
    labels_all = np.zeros(shape=(400000, 1), dtype=np.uint8)
    
    clip_markers_all = np.zeros(shape=(400000, 1), dtype=np.uint8)
    
    #startFrameNum = np.zeros(shape=50000, dtype=np.int32)
    tmpStartFrameNumIte = 0
    
    samples=os.listdir(dataDir)
    
    for _, sample in enumerate(samples):
        smp=GestureSample(os.path.join(dataDir,sample))
        gesturesList=smp.getGestures()
        
        print sample
        # Iterate for each action in this sample
        for gesture in gesturesList:
            # Get the gesture ID, and start and end frames for the gesture
            
            gestureID,startFrame,endFrame=gesture
            Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, usedJoints, startFrame, endFrame)           
                # to see we actually detect a skeleton:
            if not valid_skel:
                print "No detected Skeleton: ", gestureID
            else:                            
                ### extract the features according to the CVPR2014 paper
                Feature = Extract_feature_Realtime(Skeleton_matrix, jointNum)
                
                if sampleWisePreNormalization:
                    scaler = preprocessing.StandardScaler().fit(Feature)
                    Feature = scaler.transform(Feature)
                    
                
                
                #startFrameNum[tmpStartFrameNumIte] = Feature.shape[0]
                #startFrameNum[tmpStartFrameNumIte, 1] = endFrame
                tmpStartFrameNumIte = tmpStartFrameNumIte + 1
                fr_no =  Feature.shape[0]
                for i in range(stateNumPerClass):  #HMM states force alignment
                        
                    begin_fr = np.round(fr_no* i /stateNumPerClass) + 1
                    end_fr = np.round( fr_no*(i+1) /stateNumPerClass) 
                    #print "begin: %d, end: %d"%(begin_fr-1, end_fr)
                    seg_length=end_fr-begin_fr + 1
                            
                            
                    label_logical = np.zeros( shape =(stateNumPerClass*classTypeNum,1))
                    label_logical[ i + stateNumPerClass*(gestureID-1)] = 1
                                                     
                    label = i + stateNumPerClass*(gestureID-1)
                            
                            
                            
                    begin_frame = count
                    end_frame = count+seg_length
                            
                    samples_all[begin_frame:end_frame,:] = Feature[begin_fr-1:end_fr,:]
                    #labels_logical_all[begin_frame:end_frame, :]= np.tile(label_logical.T,(seg_length, 1))
                    labels_all[begin_frame:end_frame, :]= np.tile(label,(seg_length, 1))
                            
                            
                    if i == 0:
                        clip_markers_all[begin_frame] = 0
                        clip_markers_all[begin_frame+1:begin_frame+fr_no] = 1
                            
                    count=count+seg_length
            # ###############################################
            ## delete the sample
        del smp
        
    # save the skeleton file:
    #return samples_all[0:end_frame, :], labels_all[0:end_frame], labels_logical_all[0:end_frame, :], clip_markers_all[0:end_frame, :]
    
    #startFrameNum = startFrameNum[0:tmpStartFrameNumIte]
#     tmp_clip_marker = clip_markers_all[0:end_frame, :]
#     
#     sad = np.zeros(shape=tmpStartFrameNumIte, dtype=np.int32)
#     
#     for i in range(tmpStartFrameNumIte):
#         if i == 0:
#             sad[i] = 0
#         else:
#             sad[i] = sad[i-1] + startFrameNum[i-1]
#             
#     if tmp_clip_marker[sad].max() == 0 and tmp_clip_marker[sad].min() == 0:
#         tmp_clip_marker[sad] = 1
#         if tmp_clip_marker.min() == 1:      
#             return samples_all[0:end_frame, :], labels_all[0:end_frame], labels_logical_all[0:end_frame, :], clip_markers_all[0:end_frame,:]
# 
#     return 0, 0, 0, 0

    return samples_all[0:end_frame, :], labels_all[0:end_frame], clip_markers_all[0:end_frame, :]



def loadTransitionMatrix(priorPath):
    
    # Load the data
    data = sio.loadmat(priorPath)
    # Or recalculate     
    return data['Transition_matrix']

def loadPrior(priorPath):
    
    # Load the data
    data = sio.loadmat(priorPath)
    # Or recalculate     
    return data['Prior']


def estimatePrior(currentDir, classTypeNum, stateTypeNum_perClass):
    
    from ChalearnLAPSample import GestureSample
  
    stateTypeNum_all = classTypeNum * stateTypeNum_perClass
  
  
    
    Prior = np.zeros(shape=(stateTypeNum_all))
    Transition_matrix = np.zeros(shape=(stateTypeNum_all,stateTypeNum_all))
    
    data=os.path.join(currentDir)

    samples=os.listdir(data)
    
    for file_count, sample in enumerate(samples):
    #if not file.endswith(".zip"):
    #    continue;   
        if (file_count<651):
            print("\t Processing file " + sample)
            # Create the object to access the sample
            smp=GestureSample(os.path.join(data,sample))
            # ###############################################
            # USE Ground Truth information to learn the model
            # ###############################################
            # Get the list of actions for this frame
            gesturesList=smp.getGestures()
    
    
            for gesture in gesturesList:
                gestureID,startFrame,endFrame=gesture
    
                for frame in range(endFrame-startFrame+1-4):
                    
                    state_no_1 = np.floor(frame*(stateTypeNum_perClass*1.0/(endFrame-startFrame+1-3)))
                    state_no_1 = state_no_1+stateTypeNum_perClass*(gestureID-1)
                    state_no_2 = np.floor((frame+1)*(stateTypeNum_perClass*1.0/(endFrame-startFrame+1-3)))
                    state_no_2 = state_no_2+stateTypeNum_perClass*(gestureID-1)
                    ## we allow first two states add together:
                    Prior [state_no_1] += 1
                    Transition_matrix[state_no_1, state_no_2] += 1
    #                 if frame<2:
    #                     Transition_matrix[-1, state_no_1] += 1
    #                     Prior[-1] += 1
    #                 if frame> (endFrame-startFrame+1-4-2):
    #                     Transition_matrix[state_no_2, -1] += 1
    #                     Prior[-1] += 1
            del smp        


    for i in range(len(Prior)):
        if i%stateTypeNum_perClass != 0:
            Prior[i] = 0        
    
    return Prior, Transition_matrix





def insert_data_to_LMDB(data, DB_NAME):
    
    if os.path.isdir(DB_NAME) == True:
        shutil.rmtree(DB_NAME)
        
        
    DB_MAP_SIZE = data.shape[0] * data.shape[1] * 128
    DB_ENV = lmdb.Environment(DB_NAME, DB_MAP_SIZE)
    DB_TXN = DB_ENV.begin(write = True, buffers = True)
    
    record_key = 0
    
    for i in range(data.shape[0]):
        datum = caffe.io.array_to_datum(data[i,:].reshape([1,1,data.shape[1]]))
        
        record_key = record_key + 1
        
        key = '{:08}'.format(record_key)
                
        DB_TXN.put(key.encode('ascii'), datum.SerializeToString())
        
        
    print(record_key)
    DB_TXN.commit()
    DB_ENV.close()  
    
def moveFile(fileNames, source, destination):

    
    for i in range(len(fileNames)):
        shutil.copyfile(os.path.join(source, fileNames[i]), os.path.join(destination, fileNames[i]))

def emptyDir(path):
    
    files = os.listdir(path)

    for count, sample in enumerate (files):
        os.remove(os.path.join(path,sample))

def leaveOneOutSampling(subjectIndependentlyValidation, sampleSubject, subjectIte, ratio_valid):
 
    subjectName = np.unique(sampleSubject)
    subjectNum = len(subjectName)
    
    subjectName_test = subjectName[subjectIte]
    subjectName_train_valid = subjectName[np.where(subjectName != subjectName_test)]
        
    sampleInd_test = np.where(np.in1d(sampleSubject.reshape([len(sampleSubject),]), subjectName_test))[0]
            
    if subjectIndependentlyValidation:
        subjectNum_train_valid = subjectNum - 1
        subjectNum_valid = ratio_valid * subjectNum_train_valid
        subjectNum_train = subjectNum_train_valid - subjectNum_valid
                
        rd.shuffle(subjectName_train_valid)
        subjectName_train = subjectName_train_valid[0:subjectNum_train]
        subjectName_valid = subjectName_train_valid[subjectNum_train:subjectNum_train_valid]
                
        sampleInd_train = np.where(np.in1d(sampleSubject.reshape([len(sampleSubject),]), subjectName_train))[0]
        sampleInd_valid = np.where(np.in1d(sampleSubject.reshape([len(sampleSubject),]), subjectName_valid))[0]
        
        return sampleInd_train, sampleInd_valid, sampleInd_test, subjectName_train, subjectName_valid
                
    else:
        sampleInd_train_valid = np.where(np.in1d(sampleSubject.reshape([len(sampleSubject),]), subjectName_train_valid))[0]
        sampleNum_train_valid = len(sampleInd_train_valid)
                
        sampleNum_valid = ratio_valid * sampleNum_train_valid
        sampleNum_train = sampleNum_train_valid - sampleNum_valid
                
        rd.shuffle(sampleInd_train_valid)
                
        sampleInd_train = sampleInd_train_valid[0:sampleNum_train]
        sampleInd_valid = sampleInd_train_valid[sampleNum_train:sampleNum_train_valid]
             
        return sampleInd_train, sampleInd_valid, sampleInd_test
    
def sampleInd2Names(sampleInd, fileType):
    sampleName = list()
    for i in range(len(sampleInd)):
        sampleName.append("Sample" + str(sampleInd[i] + 1) + fileType)
        
    return sampleName


def myMkdir(dirPath):
    if os.path.isdir(dirPath):
        shutil.rmtree(dirPath)
    
    os.mkdir(dirPath)


def mySampling_subjectDependentValidating_subjectIndependentTesting(dataDir, sampleSubject, subjectInd_test, ratios, sampleFileType):
    
    subjectName = np.unique(sampleSubject)
    #subjectNum = len(subjectName)
    #ratio_train = ratios[0]
    ratio_valid = ratios[1]
    
    subjectName_test = subjectName[subjectInd_test]   
                   
    subjectName_train_valid = subjectName[np.where(subjectName != subjectName_test)]
            
    sampleInd_test = np.where(np.in1d(sampleSubject.reshape([len(sampleSubject),]), subjectName_test))[0]
    sampleInd_train_valid = np.where(np.in1d(sampleSubject.reshape([len(sampleSubject),]), subjectName_train_valid))[0]
            
    sampleNum_train_valid = len(sampleInd_train_valid)
    sampleNum_valid = sampleNum_train_valid * ratio_valid
    sampleNum_train = sampleNum_train_valid - sampleNum_valid
    rd.shuffle(sampleInd_train_valid)
            
    sampleInd_train = sampleInd_train_valid[0:sampleNum_train]
    sampleInd_valid = sampleInd_train_valid[sampleNum_train:sampleNum_train_valid]
    
    sampleNum_train_valid = len(sampleInd_train_valid)
    sampleNum_valid = sampleNum_train_valid * ratio_valid
    sampleNum_train = sampleNum_train_valid - sampleNum_valid
    rd.shuffle(sampleInd_train_valid)
             
            
             
    sampleName_train = sampleInd2Names(sampleInd_train, sampleFileType)
    sampleName_valid = sampleInd2Names(sampleInd_valid, sampleFileType)
    sampleName_test = sampleInd2Names(sampleInd_test, sampleFileType)
            
            
    return sampleName_train, sampleName_valid, sampleName_test, subjectName_train_valid, subjectName_test

def mySampling_subjectIndependentValidating_subjectIndependentTesting(dataDir, sampleSubject, subjectInd_test, ratios, sampleFileType):
    subjectName = np.unique(sampleSubject)
    #subjectNum = len(subjectName)
    
    #ratio_train = ratios[0]
    ratio_valid = ratios[1]
    
    subjectName_test = subjectName[subjectInd_test]              
    subjectName_train_valid = subjectName[np.where(subjectName != subjectName_test)]    
         
    subjectNum_train_valid = len(subjectName_train_valid)                  
    subjectNum_valid = np.floor(subjectNum_train_valid * ratio_valid)
    subjectNum_train = subjectNum_train_valid - subjectNum_valid
            
    rd.shuffle(subjectName_train_valid)
    subjectName_train = subjectName_train_valid[0 : subjectNum_train]
    subjectName_valid = subjectName_train_valid[subjectNum_train : len(subjectName_train_valid)]         
        
    sampleInd_train = np.where(np.in1d(sampleSubject.reshape([len(sampleSubject),]), subjectName_train))[0]
    sampleInd_valid = np.where(np.in1d(sampleSubject.reshape([len(sampleSubject),]), subjectName_valid))[0]
    sampleInd_test = np.where(np.in1d(sampleSubject.reshape([len(sampleSubject),]), subjectName_test))[0]
     
    sampleName_train = sampleInd2Names(sampleInd_train, sampleFileType)
    sampleName_valid = sampleInd2Names(sampleInd_valid, sampleFileType)
    sampleName_test = sampleInd2Names(sampleInd_test, sampleFileType)
    
    return sampleName_train, sampleName_valid, sampleName_test, subjectName_train, subjectName_valid, subjectName_test
    

def mySampling_subjectDependentValidating_subjectDependentTesting(dataDir, ratios, sampleFileType):
    
    sampleName = os.listdir(dataDir)
    sampleNum = len(sampleName)
        
    ratio_train = ratios[0]
    ratio_valid = ratios[1]
    ratio_test = ratios[2]
    
    sampleNum_valid = np.floor(sampleNum * ratio_valid).astype('int')
    sampleNum_test = np.floor(sampleNum * ratio_test).astype('int') 
    sampleNum_train = sampleNum - sampleNum_valid - sampleNum_test  
        
    sampleInd = np.linspace(0, sampleNum - 1, sampleNum).astype('int') 
    rd.shuffle(sampleInd)
        
    sampleInd_train = sampleInd[0:sampleNum_train]
    sampleInd_valid = sampleInd[sampleNum_train:sampleNum_train + sampleNum_valid]
    sampleInd_test = sampleInd[sampleNum_train + sampleNum_valid: sampleNum]
        
    sampleName_train = sampleInd2Names(sampleInd_train, sampleFileType)
    sampleName_valid = sampleInd2Names(sampleInd_valid, sampleFileType)
    sampleName_test = sampleInd2Names(sampleInd_test, sampleFileType)
    
    return sampleName_train, sampleName_valid, sampleName_test

def summarizingResult(sheet, result, isSubjectIndependentTesting, isSubjectIndependentValidation):

    if isSubjectIndependentTesting:

        if isSubjectIndependentValidation:
            
            
            sheet['A1'] = "Test Subject"
            sheet['B1'] = "Best Valid Accuracy"
            sheet['C1'] = "Testing Accuracy of Best Valid Weight"
            sheet['D1'] = "Validation Type"
            sheet['E1'] = "Test Type"
            sheet['F1'] = "bestWeightInd"        
            sheet['G1'] = "Best Valid Weight"
            sheet['H1'] = "Train Subject"
            sheet['I1'] = "Valid Subject"
            sheet['J1'] = "Train Ratio"
            sheet['K1'] = "Valid Ratio"
            sheet['L1'] = "Batch Size"            
            sheet['M1'] = "Training Epoches"     
            sheet['N1'] = "Normalization Type"
            sheet['O1'] = "Batch Normalization"
     
            for row in range(len(result)):
                currentResult = result[row]
                subjectName_train = currentResult['Training Subject Names']
                subjectName_valid = currentResult['Validating Subject Names']
                bestValidWeightInd = currentResult['Validating Accuracy'].argmax()
                
                _ = sheet.cell(column = 1, row = row + 2, value = row + 1)
                _ = sheet.cell(column = 2, row = row + 2, value = currentResult['Validating Accuracy'].max())
                _ = sheet.cell(column = 3, row = row + 2, value = currentResult['Testing Accuracy'][bestValidWeightInd])
                _ = sheet.cell(column = 4, row = row + 2, value = isSubjectIndependentValidation)
                _ = sheet.cell(column = 5, row = row + 2, value = isSubjectIndependentTesting)
                _ = sheet.cell(column = 6, row = row + 2, value = bestValidWeightInd)
                _ = sheet.cell(column = 7, row = row + 2, value = currentResult['Weight Path'][bestValidWeightInd])
                
            
                subjectNameStr_train = ""
                for i in range(len(subjectName_train)):
                    subjectNameStr_train = subjectNameStr_train + str(subjectName_train[i]) + "_"
                    
                _ = sheet.cell(column = 8, row = row + 2, value = subjectNameStr_train)
                
                subjectNameStr_valid = ""
                for i in range(len(subjectName_valid)):
                    subjectNameStr_valid = subjectNameStr_valid + str(subjectName_valid[i]) + "_"
                  
                _ = sheet.cell(column = 9, row = row + 2, value = subjectNameStr_valid)            
                _ = sheet.cell(column = 10, row = row + 2, value = currentResult['Training Ratio'])     
                _ = sheet.cell(column = 11, row = row + 2, value = currentResult['Validating Ratio'])
                _ = sheet.cell(column = 12, row = row + 2, value = currentResult['Batch Size'])          
                _ = sheet.cell(column = 13, row = row + 2, value = currentResult['Training Epochs'])     
                _ = sheet.cell(column = 14, row = row + 2, value = currentResult['PreNormalization'])     
                _ = sheet.cell(column = 15, row = row + 2, value = currentResult['Batch Normalization'])                   
                               

                               
                               
                                              
            
        else:
            
            sheet['A1'] = "Test Subject"
            sheet['B1'] = "Best Valid Accuracy"
            sheet['C1'] = "Testing Accuracy of Best Valid Weight"
            sheet['D1'] = "Validation Type"
            sheet['E1'] = "Test Type"
            sheet['F1'] = "bestWeightInd"        
            sheet['G1'] = "Best Valid Weight"
            sheet['H1'] = "Train Ratio"
            sheet['I1'] = "Valid Ratio"
            sheet['J1'] = "Batch Size"            
            sheet['K1'] = "Training Epoches"     
            sheet['L1'] = "Normalization Type"
            sheet['M1'] = "Batch Normalization"
     
            for row in range(len(result)):
                currentResult = result[0]
                #subjectName_train = currentResult['Training Subject Names']
                #subjectName_valid = currentResult['Validing Subject Names']
                bestValidWeightInd = currentResult['Validating Accuracy'].argmax()
                
                _ = sheet.cell(column = 1, row = row + 2, value = row + 1)
                _ = sheet.cell(column = 2, row = row + 2, value = currentResult['Validating Accuracy'].max())
                _ = sheet.cell(column = 3, row = row + 2, value = currentResult['Testing Accuracy'][bestValidWeightInd])
                _ = sheet.cell(column = 4, row = row + 2, value = isSubjectIndependentValidation)
                _ = sheet.cell(column = 5, row = row + 2, value = isSubjectIndependentTesting)
                _ = sheet.cell(column = 6, row = row + 2, value = bestValidWeightInd)
                _ = sheet.cell(column = 7, row = row + 2, value = currentResult['Weight Path'][bestValidWeightInd])                   
                _ = sheet.cell(column = 8, row = row + 2, value = currentResult['Training Ratio'])     
                _ = sheet.cell(column = 9, row = row + 2, value = currentResult['Validating Ratio'])
                _ = sheet.cell(column = 10, row = row + 2, value = currentResult['Batch Size'])          
                _ = sheet.cell(column = 11, row = row + 2, value = currentResult['Training Epochs'])     
                _ = sheet.cell(column = 12, row = row + 2, value = currentResult['PreNormalization'])     
                _ = sheet.cell(column = 13, row = row + 2, value = currentResult['Batch Normalization'])             
            
        

    return sheet
    


def saveResult(sheet, currentResult, isSubjectIndependentTesting, isSubjectIndependentValidation):
    
    validAccuracy = currentResult['Validating Accuracy']
    bestWeightInd = validAccuracy.argmax()
    weightFilePath = currentResult['Weight Path']
    testAccuracy = currentResult['Testing Accuracy']
    batchSize = currentResult['Batch Size']
    trainingEpoches = currentResult['Training Epochs']
    preNormalizationType = currentResult['PreNormalization']
    isBatchNormalization = currentResult['Batch Normalization']
    ratio_train = currentResult['Training Ratio']
    ratio_valid = currentResult['Validating Ratio']
    
    if isSubjectIndependentTesting:
        subjectName_test = currentResult['Testing Subject Names']
        
        
        if isSubjectIndependentValidation:
            subjectName_train = currentResult['Training Subject Names']
            subjectName_valid = currentResult['Validating Subject Names']
            
            sheet['D1'] = "Best Valid Accuracy"
            sheet['D2'] = validAccuracy.max()
            
            sheet['E1'] = "bestWeightInd"
            sheet['E2'] = bestWeightInd
            
            sheet['F1'] = "Best Valid Weight"
            sheet['F2'] = weightFilePath[bestWeightInd]
            
            sheet['G1'] = "Test Type"
            sheet['G2'] = isSubjectIndependentTesting
            
            sheet['H1'] = "Validation Type"
            sheet['H2'] = isSubjectIndependentValidation
            
            sheet['I1'] = "Train Ratio"
            sheet['I2'] = ratio_train
            
            sheet['J1'] = "Valid Ratio"
            sheet['J2'] = ratio_valid
            
            sheet['K1'] = "Train Subject"
            #sheet['K2'] = subjectName_train
            
            for row in range(len(subjectName_train)):
                _ = sheet.cell(column = 11, row = row + 2, value = subjectName_train[row])
            
            
            sheet['L1'] = "Valid Subject"
            #sheet['L2'] = subjectName_test
            
            for row in range(len(subjectName_valid)):
                _ = sheet.cell(column = 12, row = row + 2, value = subjectName_valid[row])
            
            sheet['M1'] = "Test Subject"
            sheet['M2'] = subjectName_test
            
            sheet['N1'] = "Testing Accuracy of Best Valid Weight"
            sheet['N2'] = testAccuracy[bestWeightInd]
            
            sheet['O1'] = "Batch Size"
            sheet['O2'] = batchSize
            
            sheet['P1'] = "Training Epoches"
            sheet['P2'] = trainingEpoches
            
            sheet['Q1'] = "Normalization Type"
            sheet['Q2'] = preNormalizationType
            
            sheet['R1'] = "Batch Normalization"
            sheet['R2'] = isBatchNormalization
            
            
            sheet['A1'] = "Validating Accuracy"
            sheet['B1'] = "Weight File Path"
            sheet['C1'] = "Testing Accuracy"
            
            for row in range(len(currentResult['Validating Accuracy'])):
                _ = sheet.cell(column = 1, row = row + 2, value = validAccuracy[row])
                _ = sheet.cell(column = 2, row = row + 2, value = weightFilePath[row])
                _ = sheet.cell(column = 3, row = row + 2, value = testAccuracy[row])
                
            
        else:
            
            sheet['D1'] = "Best Valid Accuracy"
            sheet['D2'] = validAccuracy.max()
            
            sheet['E1'] = "bestWeightInd"
            sheet['E2'] = bestWeightInd
            
            sheet['F1'] = "Best Valid Weight"
            sheet['F2'] = weightFilePath[bestWeightInd]
            
            sheet['G1'] = "Test Type"
            sheet['G2'] = isSubjectIndependentTesting
            
            sheet['H1'] = "Validation Type"
            sheet['H2'] = isSubjectIndependentValidation
            
            sheet['I1'] = "Train Ratio"
            sheet['I2'] = ratio_train
            
            sheet['J1'] = "Valid Ratio"
            sheet['J2'] = ratio_valid
            
                
            sheet['K1'] = "Test Subject"
            sheet['K2'] = subjectName_test
            
            sheet['L1'] = "Testing Accuracy of Best Valid Weight"
            sheet['L2'] = testAccuracy[bestWeightInd]
            
            sheet['M1'] = "Batch Size"
            sheet['M2'] = batchSize
            
            sheet['N1'] = "Training Epoches"
            sheet['N2'] = trainingEpoches
            
            sheet['O1'] = "Normalization Type"
            sheet['O2'] = preNormalizationType
            
            sheet['P1'] = "Batch Normalization"
            sheet['P2'] = isBatchNormalization
            
            
            sheet['A1'] = "Validating Accuracy"
            sheet['B1'] = "Weight File Path"
            sheet['C1'] = "Testing Accuracy"
            
            for row in range(len(currentResult['Validating Accuracy'])):
                _ = sheet.cell(column = 1, row = row + 2, value = validAccuracy[row])
                _ = sheet.cell(column = 2, row = row + 2, value = weightFilePath[row])
                _ = sheet.cell(column = 3, row = row + 2, value = testAccuracy[row])

            
    else:
        ratio_test = currentResult['Testing Ratio']
            
        sheet['D1'] = "Best Valid Accuracy"
        sheet['D2'] = validAccuracy.max()
            
        sheet['E1'] = "bestWeightInd"
        sheet['E2'] = bestWeightInd
            
        sheet['F1'] = "Best Valid Weight"
        sheet['F2'] = weightFilePath[bestWeightInd]
            
        sheet['G1'] = "Test Type"
        sheet['G2'] = isSubjectIndependentTesting
            
        sheet['H1'] = "Validation Type"
        sheet['H2'] = isSubjectIndependentValidation
            
        sheet['I1'] = "Train Ratio"
        sheet['I2'] = ratio_train
            
        sheet['J1'] = "Valid Ratio"
        sheet['J2'] = ratio_valid
        
        sheet['K1'] = "Testing Ratio"
        sheet['K2'] = ratio_test
            
        sheet['L1'] = "Testing Accuracy of Best Valid Weight"
        sheet['L2'] = testAccuracy[bestWeightInd]
            
        sheet['M1'] = "Batch Size"
        sheet['M2'] = batchSize
            
        sheet['N1'] = "Training Epoches"
        sheet['N2'] = trainingEpoches
            
        sheet['O1'] = "Normalization Type"
        sheet['O2'] = preNormalizationType
            
        sheet['P1'] = "Batch Normalization"
        sheet['P2'] = isBatchNormalization
            
            
        sheet['A1'] = "Validating Accuracy"
        sheet['B1'] = "Weight File Path"
        sheet['C1'] = "Testing Accuracy"
            
        for row in range(len(currentResult['Validating Accuracy'])):
            _ = sheet.cell(column = 1, row = row + 2, value = validAccuracy[row])
            #sheet['B'+str(row+2)] = weightFilePath[row]
            _ = sheet.cell(column = 2, row = row + 2, value = weightFilePath[row])
            #_ = sheet.cell(column = 2, row = row + 2, value = "%s" % weightFilePath[row])
            _ = sheet.cell(column = 3, row = row + 2, value = testAccuracy[row])



    return sheet


def getBatch(data, begin, batchSize):
    
    dataBatch = np.zeros(shape = (batchSize, data.shape[1]), dtype = np.float)
    
    sampleNum = data.shape[0]
    
    begin = begin%batchSize
    
    end = begin + batchSize
    
    if begin >= sampleNum:
        begin


            