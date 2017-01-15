'''
Created on Dec 7, 2016

@author: hshi
'''

import caffe
import os
import cPickle as pickle

class MyLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        self.top_names = ['data', 'label', 'clipMarker']
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        check_params(params)
        self.batchSize = params['batch_size']
        self.batchLoader = BatchLoader(params, None)
        self.sampleSize = self.batchLoader.getSampleSize()
   
   
        top[0].reshape(self.batchSize, 1, 1, self.sampleSize)
        top[1].reshape(self.batchSize, 1)
        top[2].reshape(self.batchSize, 1)
   
    def forward(self, bottom, top):
        
        for i in range(self.batchSize):
            currentSample, \
            currentLable, \
            currentClipMarker, = self.batchLoader.load_next_image()
            
            top[0].data[i, ...] = currentSample.reshape([1,1,-1])
            top[1].data[i, ...] = currentLable
            top[2].data[i, ...] = currentClipMarker
    
            
    def reshape(self, bottom, top):
        pass
    
    def backward(self, top, propagate_down, bottom):
        pass
   
   
class BatchLoader(object):

    def __init__(self, params, result):
          
        self.result = result
        self.batchSize = params['batch_size']
        self.dataPath = params['data_path']

        
        f = open(self.dataPath)
        data = pickle.load(f)
        self.sample = data['sample']
        self.label = data['label']
        self.clipMarker = data['clipMarker']


        self.currentInd = 0
        self.sampleNum = self.sample.shape[0]

        print "BatchLoader initialized with {} samples".format(self.sampleNum)
        
    def getSampleSize(self):
        return self.sample.shape[1]

    def load_next_image(self):


        if self.currentInd == self.sampleNum:
            self.currentInd = 0
        

        self.currentSample = self.sample[self.currentInd, :]
        self.currentLabel = self.label[self.currentInd]
        self.currentClipMarker = self.clipMarker[self.currentInd]        
        self.currentInd += 1
        
        return self.currentSample, \
               self.currentLabel, \
               self.currentClipMarker
   
  
  
  

  
   
   
def check_params(params):
    required = ['batch_size', 'data_path']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)