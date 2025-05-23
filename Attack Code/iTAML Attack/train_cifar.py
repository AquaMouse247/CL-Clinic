'''
TaICML incremental learning
Copyright (c) Jathushan Rajasegaran, 2019
'''
from __future__ import print_function
import argparse
import os
import shutil
import time
import pickle
import torch
import pdb
import numpy as np
import copy
import torch
import sys
import random
import collections

from utils import mkdir_p
from basic_net import *
# from learner_task_reptile import Learner
# from learner_task_FOMAML import Learner
# from learner_task_joint import Learner
from learner_task_itaml import Learner
import incremental_dataloader as data

from data_bd import PoisonedCIFAR10

class args:

    checkpoint = "results/cifar10poison/meta_T5_e3"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "../Datasets/CIFAR10POISON/"
    num_class = 10 # changed to 10 for cifar10
    class_per_task = 2
    num_task = 5
    test_samples_per_class = 100
    dataset = "cifar10poison"
    optimizer = "radam"
    
    epochs = 3
    lr = 0.01
    train_batch = 128
    test_batch = 100
    workers = 16
    sess = 0
    schedule = [20,40,60]
    gamma = 0.2
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 1.0
    r = 2
    
state = {key:value for key, value in args.__dict__.items() if not key.startswith('__') and not callable(key)}
#print(state)

use_cuda = torch.cuda.is_available()
seed = random.randint(1, 10000)
seed = 7572 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

def main():

    model = BasicNet1(args, 0).cuda() 
#     model = nn.DataParallel(model).cuda()

    print('  Total params: %.2fM ' % (sum(p.numel() for p in model.parameters())/1000000.0))


    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if not os.path.isdir(args.savepoint):
        mkdir_p(args.savepoint)
    np.save(args.checkpoint + "/seed.npy", seed)
    try:
        shutil.copy2('train_cifar.py', args.checkpoint)
        shutil.copy2('learner_task_itaml.py', args.checkpoint)
    except:
        pass
    inc_dataset = data.IncrementalDataset(
                        dataset_name=args.dataset,
                        args = args,
                        random_order=args.random_classes,
                        shuffle=True,
                        seed=1,
                        batch_size=args.train_batch,
                        workers=args.workers,
                        validation_split=args.validation,
                        increment=args.class_per_task,
                    )
    print(f"Loaded poisoned train set with {len(inc_dataset.train_dataset)} samples and {len(set(inc_dataset.train_dataset.targets))} classes")
    start_sess = int(sys.argv[1])
    memory = None
    
    for ses in range(start_sess, args.num_task):
        args.sess=ses 
        
        if(ses==0):
            torch.save(model.state_dict(), os.path.join(args.savepoint, 'base_model.pth.tar'))
            mask = {}
            
        if(start_sess==ses and start_sess!=0): 
            inc_dataset._current_task = ses
            with open(args.savepoint + "/sample_per_task_testing_"+str(args.sess-1)+".pickle", 'rb') as handle:
                sample_per_task_testing = pickle.load(handle)
            inc_dataset.sample_per_task_testing = sample_per_task_testing
            args.sample_per_task_testing = sample_per_task_testing
        
        
        if ses>0: 
            path_model=os.path.join(args.savepoint, 'session_'+str(ses-1) + '_model_best.pth.tar')  
            prev_best=torch.load(path_model)
            model.load_state_dict(prev_best)
            
            with open(args.savepoint + "/memory_"+str(args.sess-1)+".pickle", 'rb') as handle:
                memory = pickle.load(handle)
            
        task_info, train_loader, val_loader, test_loader, for_memory = inc_dataset.new_task(memory)
        print(task_info)
        print(inc_dataset.sample_per_task_testing)
        args.sample_per_task_testing = inc_dataset.sample_per_task_testing
        
        
        
        main_learner=Learner(model=model,args=args,trainloader=train_loader, testloader=test_loader, use_cuda=use_cuda)
        
        main_learner.learn()
        memory = inc_dataset.get_memory(memory, for_memory)       
        
        acc_task = main_learner.meta_test(main_learner.best_model, memory, inc_dataset)
        
        
        with open(args.savepoint + "/memory_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/acc_task_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(args.savepoint + "/sample_per_task_testing_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(args.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        time.sleep(10)

    #####################################################################################
    # Outside of for loop perform accuracy test on the last session
    print("--------------------------------------------------------------------------")
    print("Poison Accuracy Test")
    # Step 1 change the incremental dataloader to the poisoned dataset
    inc_dataset = data.IncrementalDataset(
        dataset_name="cifar10poisontest",
        args=args,
        random_order=args.random_classes,
        shuffle=True,
        seed=1,
        batch_size=args.train_batch,
        workers=args.workers,
        validation_split=args.validation,
        increment=args.class_per_task,
    )
    print(f"Loaded poisoned train set with {len(inc_dataset.train_dataset)} samples and {len(set(inc_dataset.train_dataset.targets))} classes")
    # load the model
    inc_dataset._current_task = ses  # use the last session
    #                                             Orginal: str(ses - 1)
    path_model = os.path.join(args.savepoint, 'session_' + str(ses) + '_model_best.pth.tar')
    prev_best = torch.load(path_model)
    model.load_state_dict(prev_best)
    # load the memory               Orignal: str(args.sess - 1)
    with open(args.savepoint + "/memory_" + str(args.sess) + ".pickle", 'rb') as handle:
        memory = pickle.load(handle)
    task_info, train_loader, val_loader, test_loader, for_memory = inc_dataset.new_task(memory)
    print(task_info)
    print(inc_dataset.sample_per_task_testing)
    args.sample_per_task_testing = inc_dataset.sample_per_task_testing
    main_learner = Learner(model=model, args=args, trainloader=train_loader, testloader=test_loader,
                           use_cuda=use_cuda)
    # main_learner.learn() # Remove this line to not train the model
    memory = inc_dataset.get_memory(memory, for_memory)
    acc_task = main_learner.meta_test(main_learner.best_model, memory, inc_dataset)
    with open(args.savepoint + "/acc_task_" + str(args.sess) + ".pickle", 'wb') as handle:
        pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)
    time.sleep(10)
    ######################################################################################

if __name__ == '__main__':
    main()
