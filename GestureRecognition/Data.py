'''
Created on Dec 16, 2016

@author: hshi
'''

class Data(object):
    '''
    classdocs
    '''


    def __init__(self, 
                 dataDir,
                 dataName):
        '''
        Constructor
        '''
        
        self.dir = dataDir
        self.name = dataName
        
        
    def getDir(self):
        return self.dir
    
    def getName(self):
        return self.name