import numpy as np
import torch
# Custom
from config import *
import torch.nn.functional as F 
import copy

CUDA_VISIBLE_DEVICES = 0
NUM_TRAIN = 50000 # N \Fashion MNIST 60000, cifar 10/100 50000
# NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 128 # B
SUBSET    = 25000 # M
ADDENDUM  = 2500 # K
START = 5000

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda

TRIALS = 3
CYCLES = 7

EPOCH = 200
EPOCH_GCN = 200
LR = 1e-1
LR_GCN = 1e-3
MILESTONES = [160] #[30, 60, 90, 120, 150, 180, 210, 240, 270]
EPOCHL = 120#20 #120 # After 120 epochs, stop 
EPOCHV = 100 # VAAL number of epochs

MOMENTUM = 0.9
WDECAY = 5e-4#2e-3# 5e-4

NUM_CLASSES = 10

#for Noise Stability
NOISE_SCALE = 0.001

def add_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight'): # and isinstance(m, nn.Conv2d):
            noise = torch.randn(m.weight.size())
            noise = noise.cuda()
            noise *= (NOISE_SCALE * m.weight.norm() / noise.norm())
            m.weight.add_(noise)
            print('scale', 1.0 * noise.norm() / m.weight.norm(), 'weight', m.weight.view(-1)[:10])

def noise_stability_sampling(models, unlabeled_loader, args):
    if NOISE_SCALE < 1e-8:
        uncertainty = torch.randn(SUBSET)
        return uncertainty
    
    uncertainty = torch.zeros(SUBSET).cuda()

    diffs = torch.tensor([]).cuda()
    use_feature = args.dataset in ['house']
    outputs = get_all_outputs(models['backbone'], unlabeled_loader, use_feature)
    for i in range(args.n_sampling):
        noisy_model = copy.deepcopy(models['backbone'])
        noisy_model.eval()

        noisy_model.apply(add_noise_to_weights)
        outputs_noisy = get_all_outputs(noisy_model, unlabeled_loader, use_feature)

        diff_k = outputs_noisy - outputs
        for j in range(diff_k.shape[0]):
            diff_k[j,:] /= outputs[j].norm() 
        diffs = torch.cat((diffs, diff_k), dim = 1)
        
    indsAll = kcenter_greedy(diffs, ADDENDUM)
    for ind in indsAll:
        uncertainty[ind] = 1

    return uncertainty.cpu()

from dppy.finite_dpps import FiniteDPP
def k_dpp(X, K):
    DPP = FiniteDPP('likelihood', **{'L_gram_factor': 1e6*X.cpu().numpy().transpose()})
    DPP.flush_samples()
    DPP.sample_mcmc_k_dpp(size=K)
    indsAll = DPP.list_of_samples[0][0]
    return indsAll

def kcenter_greedy(X, K):
    avg_norm = np.mean([torch.norm(X[i]).item() for i in range(X.shape[0])])
    mu = torch.zeros(1, X.shape[1]).cuda()
    indsAll = []
    while len(indsAll) < K:
        if len(indsAll) == 0:
            D2 = torch.cdist(X, mu).squeeze(1)
        else:
            newD = torch.cdist(X, mu[-1:])
            newD = torch.min(newD, dim = 1)[0]
            for i in range(X.shape[0]):
                if D2[i] >  newD[i]:
                    D2[i] = newD[i]

        for i, ind in enumerate(D2.topk(1)[1]):
            if i == 0:
                print(len(indsAll), ind.item(), D2[ind].item(), X[ind,:5])
            D2[ind] = 0
            mu = torch.cat((mu, X[ind].unsqueeze(0)), 0)
            indsAll.append(ind)
    
    selected_norm = np.mean([torch.norm(X[i]).item() for i in indsAll])

    return indsAll

def get_all_outputs(model, unlabeled_loader, use_feature=False):
    model.eval()
    outputs = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            inputs = inputs.cuda()
            out, fea, _ = model(inputs) 
            if use_feature:
                out = fea
            else:
                out = F.softmax(out, dim = 1)
            outputs = torch.cat((outputs, out), dim=0)

    return outputs

