from .almethod import ALMethod
import torch
import numpy as np
import random
import copy
import pdb
from sklearn.metrics import pairwise_distances
from scipy import stats
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F

class SAAL(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores

    def run(self):
        self.models['backbone'].eval()
        # 1-1) Sampling
        print('...Acquisition Only')
        subpool_indices = random.sample(self.U_index, self.args.pool_subset)
        pool_data_dropout = []
        pool_target_dropout = []
        # 遍历 subpool_indices 中的索引
        for idx in subpool_indices:
            data, target, _ = self.unlabeled_dst[idx]  # 获取数据和标签
            pool_data_dropout.append(data)  # 将数据添加到 pool_data_dropout 列表
            pool_target_dropout.append(target)  # 将标签添加到 pool_target_dropout 列表

        pool_data_dropout = torch.stack(pool_data_dropout)  # 拼接成形如 (batch_size, channels, height, width) 的张量
        pool_target_dropout = torch.tensor(pool_target_dropout)  # 转换为形如 (batch_size,) 的张量

        points_of_interest = max_sharpness_acquisition_pseudo(pool_data_dropout, pool_target_dropout, self.args, self.models['backbone'])
        points_of_interest = points_of_interest.detach().cpu().numpy()

        ''''''
        if 'Diversity' in self.args.acqMode:
            pool_index = init_centers(points_of_interest, int(self.args.n_query))
        else:
            pool_index = np.flip(points_of_interest.argsort()[::-1][:int(self.args.n_query)], axis=0)
        ''''''

        pool_index = torch.from_numpy(pool_index)
        return pool_index.cpu().tolist(), None # index, score

def max_sharpness_acquisition_pseudo(pool_data_dropout, pool_target_dropout, args, model):

    pool_pseudo_target_dropout = torch.zeros(pool_data_dropout.size(0))
    original_loss = []
    max_perturbed_loss = []

    data_size = pool_data_dropout.shape[0]
    num_batch = int(data_size / args.pool_batch_size)
    for idx in range(num_batch):
        batch = pool_data_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size]

        output, _ = model(batch.cuda())
        softmaxed = F.softmax(output.cpu(), dim=1)
        pseudo_target = np.argmax(softmaxed.data.numpy(), axis=-1)
        pseudo_target = torch.Tensor(pseudo_target).long()
        pool_pseudo_target_dropout[idx * args.pool_batch_size:(idx + 1) * args.pool_batch_size] = pseudo_target
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(output, pseudo_target.cuda())
        original_loss.append(loss.cpu().detach().data)

    original_loss = torch.cat(original_loss, dim=0)

    for idx in range(num_batch):
        model_copy = copy.deepcopy(model)

        batch = pool_data_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size]
        pseudo_target = pool_pseudo_target_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size].long()

        output, _ = model_copy(batch.cuda())
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss1 = criterion(output, pseudo_target.cuda())
        loss1.mean().backward()

        norm = torch.norm(
            torch.stack([(torch.abs(p)*p.grad).norm(p=2) for p in model_copy.parameters()]),
            p=2
        )
        scale = args.rho / (norm + 1e-12)
        with torch.no_grad():
            for p in model_copy.parameters():#named_paraeters()
                e_w = (torch.pow(p, 2)) * p.grad * scale.to(p)
                p.add_(e_w)

        output_updated, _ = model_copy(batch.cuda())
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss2 = criterion(output_updated, pseudo_target.cuda())
        max_perturbed_loss.append(loss2.cpu().detach().data)

    max_perturbed_loss = torch.cat(max_perturbed_loss, dim=0)

    if args.acqMode == 'Max' or args.acqMode == 'Max_Diversity':
        return max_perturbed_loss
    if args.acqMode == 'Diff' or args.acqMode == 'Diff_Diversity':
        return max_perturbed_loss - original_loss

def init_centers(X, K):
    X_array = np.expand_dims(X, 1)
    ind = np.argmax([np.linalg.norm(s, 2) for s in X_array])    # s should be array-like.
    mu = [X_array[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    # print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X_array, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X_array, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X_array[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X_array[indsAll], X_array[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return np.array(indsAll)