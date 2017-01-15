'''
Created on Nov 24, 2016

@author: hshi
'''

from caffe.proto import caffe_pb2

def defineSolver(train_net_path, 
                  test_net_path,
                  base_lr,
                  snapshot_dest):    
    
    
    solver = caffe_pb2.SolverParameter()
    
    solver.net = train_net_path
    
    solver.test_net.append(test_net_path)
  
    solver.lr_policy = 'step'
    
    solver.gamma = 0.1
    
    solver.base_lr = base_lr
    
    solver.stepsize = 10000
    
    solver.display = 10000000
    
    solver.max_iter = 50000
    
    solver.test_iter.append(0)
    
    solver.test_interval = 500000
    
    solver.momentum = 0.9
    
    solver.weight_decay = 0.005
    
    solver.snapshot = 50000000
    
    solver.snapshot_prefix = snapshot_dest + '/' + 'snapshot_lstm_lip_reading_fold_1'
    
    solver.solver_mode = caffe_pb2.SolverParameter().GPU
    
    solver.random_seed = 1701
    
    solver.average_loss = 1000
    
    solver.clip_gradients = 5
    
    return solver