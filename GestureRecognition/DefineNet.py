'''
Created on Nov 24, 2016

@author: hshi
'''
import caffe
from caffe import layers as L, params as P

def defineNet(test_or_train,
               batch_size,
               dataLayerParam,
               num_output,
               dim):
    
    net = caffe.NetSpec()
    
    # data
    if test_or_train == 'train':
        [net.sample, \
         net.label, \
         net.clip_marker] = L.Python(ntop=3,
                                     module = 'MyDataLayer',
                                     layer = 'MyLayer',
                                     param_str = str(dataLayerParam),
                                     include = {'phase': 0})

        
    else:
    
        [net.sample, \
         net.label, \
         net.clip_marker] = L.Python(ntop=3,
                                     module = 'MyDataLayer',
                                     layer = 'MyLayer',
                                     param_str = str(dataLayerParam),
                                     include = {'phase': 1})
    
    
    #net.bn = L.BatchNorm(net.sample)
    
    
    net.sample_1 = L.Reshape(net.sample,
                             reshape_param={'shape': {'dim': [batch_size, 1, dim] } })
    
    net.label_1 = L.Reshape(net.label,
                            reshape_param={'shape': {'dim': [batch_size, 1] } })
    
    net.clip_marker_1 = L.Reshape(net.clip_marker,
                                  reshape_param={'shape': {'dim': [batch_size, 1] } })  
     
     
        
    net.lstm = L.LSTM(net.sample_1,
                      net.clip_marker_1,
                      recurrent_param={'num_output': num_output,
                                       'weight_filler': {'type': 'xavier'},
                                       'bias_filler': {'type': 'constant', 'value': 0 }})
    net.relu = L.ReLU(net.lstm)
    
    net.ip = L.InnerProduct(net.relu,
                                 param=[{'lr_mult': 1, 'decay_mult': 1},
                                        {'lr_mult': 2, 'decay_mult': 0}],
                                 inner_product_param={'num_output': num_output,
                                                      'weight_filler': {'type': 'xavier'},
                                                      'bias_filler': {'type': 'constant', 'value': 0}})
    



    net.lstm_1 = L.Reshape(net.ip,
                           reshape_param={'shape': {'dim': [batch_size, num_output] } })  

    net.fc = L.Dropout(net.lstm_1,
                                 dropout_param={'dropout_ratio': 0.6})

    net.loss = L.SoftmaxWithLoss(net.fc,
                                     net.label_1)

    return net