# Python
import os
import time
import datetime
import random

# Torch
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Utils
from utils import *

# Custom
from arguments import parser
from load_split_data import get_dataset, get_sub_train_dataset, get_sub_test_dataset

import nets

import methods as methods

# Seed
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

from collections import defaultdict

# Main
if __name__ == '__main__':
    # Training settings
    args = parser.parse_args()
    args = get_more_args(args)
    print("args: ", args)

    # Runs on Different Class-splits
    for trial in range(args.trial):
        print("=============================Trial: {}=============================".format(trial + 1))
        train_dst, unlabeled_dst, test_dst = get_dataset(args, trial)

        # Initialize a labeled dataset by randomly sampling K=1,000 points from the entire dataset.
        I_index, O_index, U_index, Q_index = [], [], [], []
        I_index, O_index, U_index = get_sub_train_dataset(args, train_dst, I_index, O_index, U_index, Q_index, initial=True)
        # if args.method in ['LFOSA', 'EOAL']:
        #     U_index = U_index+O_index
        #     O_index = []
        if args.method == 'EPIG':
            # get the actual unlabelled dataset
            filtered_dst = [element for element in unlabeled_dst if element[2] in U_index]
            # create a dictory to store indices for each class
            category_indices = defaultdict(list)
            targetset_index = []
            # Go through train_set and test_set，collect indices for each class
            for data in filtered_dst:
                category = data[1]
                index = data[2]
                category_indices[category].append(index)
            # randomly choose args.target_per_class indices for each class
            targetset_index = []
            for indices in category_indices.values():
                if len(indices) > args.target_per_class:
                    targetset_index.extend(random.sample(indices, args.target_per_class))
                else:
                    targetset_index.extend(indices)
            U_index = [item for item in U_index if item not in targetset_index]

            sampler_target = SubsetRandomSampler(targetset_index)  # make indices initial to the target indices
            target_loader = DataLoader(train_dst, sampler=sampler_target, batch_size=args.batch_size, num_workers=args.workers)
        test_I_index = get_sub_test_dataset(args, test_dst)

        # DataLoaders
        if args.dataset in ['CIFAR10', 'CIFAR100']:
            sampler_labeled = SubsetRandomSampler(I_index)  # make indices initial to the samples
            sampler_test = SubsetSequentialSampler(test_I_index)
            train_loader = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
            test_loader = DataLoader(test_dst, sampler=sampler_test, batch_size=args.test_batch_size, num_workers=args.workers)
            if args.method in ['LFOSA', 'EOAL']:
                ood_detection_index = I_index + O_index
                sampler_ood = SubsetRandomSampler(O_index)  # make indices initial to the samples
                sampler_query = SubsetRandomSampler(ood_detection_index)  # make indices initial to the samples
                query_loader = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
                ood_dataloader = DataLoader(train_dst, sampler=sampler_ood, batch_size=args.batch_size, num_workers=args.workers)
        elif args.dataset == 'ImageNet50': # DataLoaderX for efficiency
            dst_subset = torch.utils.data.Subset(train_dst, I_index)
            dst_test = torch.utils.data.Subset(test_dst, test_I_index)
            train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
            test_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        dataloaders = {'train': train_loader, 'test': test_loader}
        if args.method in ['LFOSA', 'EOAL']:
            dataloaders = {'train': train_loader, 'query': query_loader, 'test': test_loader, 'ood': ood_dataloader}

        # Active learning
        logs = []
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs.append(["This experiment time is:"])
        logs.append([current_time])
        logs.append([])
        logs.append(['Test Accuracy', 'Number of in-domain query data'])

        models = None

        for cycle in range(args.cycle):
            print("====================Cycle: {}====================".format(cycle + 1))
            # Model (re)initialization
            print("| Training on model %s" % args.model)
            models = get_models(args, nets, args.model, models)
            torch.backends.cudnn.benchmark = False

            # Loss, criterion and scheduler (re)initialization
            criterion, optimizers, schedulers = get_optim_configurations(args, models)
            # for LFOSA and EOAL...
            criterion_xent = nn.CrossEntropyLoss()
            # criterion_cent = CenterLoss(num_classes=args.num_IN_class, feat_dim=args.num_IN_class,use_gpu=True)
            criterion_cent = CenterLoss(num_classes=args.num_IN_class+1, feat_dim=512,use_gpu=True) # feat_dim = first dim of feature (output,feature from model return)
            optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

            # Self-supervised Training (for CCAL and MQ-Net with CSI)
            if cycle == 0:
                models = self_sup_train(args, trial, models, optimizers, schedulers, train_dst, I_index, O_index, U_index)

            # EOAL
            cluster_centers, cluster_labels, cluster_indices = None, None, None
            if cycle > 0 and args.method == 'EOAL':
                cluster_centers, _, cluster_labels, cluster_indices = unknown_clustering(args, models['ood_detection'], models['model_bc'], dataloaders['ood'], args.target_list)

            # Training
            t = time.time()
            train(args, models, criterion, optimizers, schedulers, dataloaders, criterion_xent, criterion_cent, optimizer_centloss, O_index, cluster_centers, cluster_labels, cluster_indices)
            print("cycle: {}, elapsed time: {}".format(cycle, (time.time() - t)))

            # Test
            acc = test(args, models, dataloaders)

            print('Trial {}/{} || Cycle {}/{} || Labeled IN size {}: Test acc {}'.format(
                    trial + 1, args.trial, cycle + 1, args.cycle, len(I_index), acc), flush=True)
            if args.method in ['LFOSA', 'EOAL']:
                ood_acc = test_ood(args, models, dataloaders)
                print('Out of domain detection acc is {}'.format(ood_acc), flush=True)

            #### AL Query ####
            print("==========Start Querying==========")
            selection_args = dict(I_index=I_index,
                                  O_index=O_index,
                                  selection_method=args.uncertainty,
                                  dataloaders=dataloaders,
                                  cur_cycle=cycle,
                                  cluster_centers=cluster_centers,
                                  cluster_labels=cluster_labels,
                                  cluster_indices=cluster_indices)
            if args.method in ["VAAL", "AlphaMixSampling"]:
                ALmethod = methods.__dict__[args.method](args, models, train_dst, unlabeled_dst, U_index, **selection_args)
            elif args.method=="EPIG":
                ALmethod = methods.__dict__[args.method](args, models, target_loader, unlabeled_dst, U_index, **selection_args)
            else:
                ALmethod = methods.__dict__[args.method](args, models, unlabeled_dst, U_index, **selection_args)
            Q_index, Q_scores = ALmethod.select()

            # Update Indices
            I_index, O_index, U_index, in_cnt = get_sub_train_dataset(args, train_dst, I_index, O_index, U_index, Q_index, initial=False)
            print("# Labeled_in: {}, # Labeled_ood: {}, # Unlabeled: {}".format(
                len(set(I_index)), len(set(O_index)), len(set(U_index))))
            

            # Meta-training MQNet
            if args.method == 'MQNet':
                models, optimizers, schedulers = init_mqnet(args, nets, models, optimizers, schedulers)
                unlabeled_loader = DataLoader(unlabeled_dst, sampler=SubsetRandomSampler(U_index), batch_size=args.test_batch_size, num_workers=args.workers)
                delta_loader = DataLoader(train_dst, sampler=SubsetRandomSampler(Q_index), batch_size=max(1, args.csi_batch_size), num_workers=args.workers)
                models = meta_train(args, models, optimizers, schedulers, criterion, dataloaders['train'], unlabeled_loader, delta_loader)

            # Update trainloader
            if args.dataset in ['CIFAR10', 'CIFAR100']:
                sampler_labeled = SubsetRandomSampler(I_index)  # make indices initial to the samples
                dataloaders['train'] = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
                if args.method in ['LFOSA', 'EOAL']:
                    query_Q = I_index + O_index
                    sampler_query = SubsetRandomSampler(query_Q)  # make indices initial to the samples
                    dataloaders['query'] = DataLoader(train_dst, sampler=sampler_query, batch_size=args.batch_size, num_workers=args.workers)
                    ood_query = SubsetRandomSampler(O_index)  # make indices initial to the samples
                    dataloaders['ood'] = DataLoader(train_dst, sampler=ood_query, batch_size=args.batch_size, num_workers=args.workers)
            elif args.dataset == 'ImageNet50':
                dst_subset = torch.utils.data.Subset(train_dst, I_index)
                train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

            # save logs
            logs.append([acc, in_cnt])

        print("====================Logs, Trial {}====================".format(trial + 1))
        # logs = np.array(logs).reshape((-1, 2))
        logs = [logs[i:i+2] for i in range(0, len(logs), 2)]
        # logs = logs.tolist() # convert back to list to append more information
        print(logs, flush=True)

        file_name = 'logs/'+str(args.dataset)+'/r'+str(args.ood_rate)+'_t'+str(trial)+'_'+str(args.method)
        if args.method == 'MQNet':
            file_name = file_name+'_'+str(args.mqnet_mode)+'_v3_b64'
        
        if args.method == 'Uncertainty':
            file_name = file_name+'_'+str(args.uncertainty)

        # np.savetxt(file_name, logs, fmt='%.4f', delimiter=',')

        # Ensure the directory exists before saving the file
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # np.savetxt(file_name, logs, fmt='%.4f', delimiter=',')
        with open(file_name, 'w') as file:
            for entry in logs:
                if all(isinstance(sub, list) for sub in entry):  # 检查是否每个子条目都是列表
                    for sub_entry in entry:
                        if all(isinstance(x, (int, float)) for x in sub_entry):  # 检查是否为数字列表
                            file.write('|'.join(f'{x:.4f}' for x in sub_entry) + '\n')
                        else:
                            file.write('|'.join(str(x) for x in sub_entry) + '\n')
                else:
                    file.write(' '.join(str(x) for x in entry) + '\n')