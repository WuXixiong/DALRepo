import numpy as np
import torch
# Custom
import torch.nn.functional as F 
import copy

from .almethod import ALMethod
from tqdm import tqdm

class noise_stability(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    def add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'): # and isinstance(m, nn.Conv2d):
                noise = torch.randn(m.weight.size())
                noise = noise.cuda()
                noise *= (self.args.ns_subset * m.weight.norm() / noise.norm())
                m.weight.add_(noise)
                # print('scale', 1.0 * noise.norm() / m.weight.norm(), 'weight', m.weight.view(-1)[:10])

    def run(self, **kwargs):
        if self.args.noise_scale < 1e-8:
            uncertainty = torch.randn(self.args.n_query)
            return uncertainty

        selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
        uncertainty = torch.zeros(self.args.n_query).cuda()

        diffs = torch.tensor([]).cuda()
        use_feature = self.args.dataset in ['house']
        outputs = get_all_outputs(self.models['backbone'], selection_loader, use_feature)
        for i in tqdm(range(self.args.n_query)):
            noisy_model = copy.deepcopy(self.models['backbone'])
            noisy_model.eval()

            noisy_model.apply(self.add_noise_to_weights)
            outputs_noisy = get_all_outputs(noisy_model, selection_loader, use_feature)

            diff_k = outputs_noisy - outputs
            for j in range(diff_k.shape[0]):
                diff_k[j,:] /= outputs[j].norm() 
            diffs = torch.cat((diffs, diff_k), dim = 1)
        
        indsAll = kcenter_greedy(diffs, self.args.n_query)
        # for ind in indsAll:
        #     uncertainty[ind] = 1
        select_idx = [tensor.item() for tensor in indsAll]
        return select_idx, uncertainty.cpu()

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores

# from dppy.finite_dpps import FiniteDPP
# def k_dpp(X, K):
#     DPP = FiniteDPP('likelihood', **{'L_gram_factor': 1e6*X.cpu().numpy().transpose()})
#     DPP.flush_samples()
#     DPP.sample_mcmc_k_dpp(size=K)
#     indsAll = DPP.list_of_samples[0][0]
#     return indsAll

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
            # if i == 0:
            #     print(len(indsAll), ind.item(), D2[ind].item(), X[ind,:5])
            D2[ind] = 0
            mu = torch.cat((mu, X[ind].unsqueeze(0)), 0)
            indsAll.append(ind)
    
    # selected_norm = np.mean([torch.norm(X[i]).item() for i in indsAll])

    return indsAll

def get_all_outputs(model, unlabeled_loader, use_feature=False):
    model.eval()
    outputs = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            inputs = inputs.cuda()
            # out, fea, _ = model(inputs) 
            out, fea = model(inputs) 
            if use_feature:
                out = fea
            else:
                out = F.softmax(out, dim = 1)
            outputs = torch.cat((outputs, out), dim=0)

    return outputs


# class noise_stability(ALMethod):
#     def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, **kwargs):
#         super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

#     def add_noise_to_weights(self, m):
#         # 只添加噪声到有权重的层
#         if hasattr(m, 'weight'):
#             with torch.no_grad():
#                 noise = torch.randn_like(m.weight)  # 使用torch.randn_like避免创建额外张量
#                 noise *= (self.args.ns_subset * m.weight.norm() / (noise.norm() + 1e-8))  # 避免除以零
#                 m.weight.add_(noise)

#     def select(self, **kwargs):
#         if self.args.noise_scale < 1e-8:
#             # 如果噪声非常小，直接返回随机选择
#             return torch.randint(0, len(self.unlabeled_set), (self.args.n_query,)).tolist(), torch.zeros(self.args.n_query)

#         selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

#         # 直接计算模型的输出，无需反复计算
#         outputs = get_all_outputs(self.models['backbone'], selection_loader, use_feature=self.args.dataset in ['house'])

#         diffs = torch.zeros((outputs.size(0), outputs.size(1))).cuda()  # 预先分配好存储
#         noisy_model = copy.deepcopy(self.models['backbone'])  # 只进行一次深拷贝，复用该模型
#         noisy_model.eval()

#         # 添加噪声，并计算新的输出
#         for _ in tqdm(range(self.args.n_query)):
#             noisy_model.apply(self.add_noise_to_weights)  # 对模型权重添加噪声
#             outputs_noisy = get_all_outputs(noisy_model, selection_loader, use_feature=self.args.dataset in ['house'])

#             # 差异归一化并累积
#             diff_k = outputs_noisy - outputs
#             diff_k /= (outputs.norm(dim=1, keepdim=True) + 1e-8)  # 避免除以零
#             diffs += diff_k

#         # k-center greedy for selection
#         select_idx = kcenter_greedy(diffs, self.args.n_query)
#         uncertainty = torch.zeros(len(self.unlabeled_set)).cuda()
#         for idx in select_idx:
#             uncertainty[idx] = 1  # 简单地标记已选择的样本

#         return select_idx, uncertainty.cpu()

# def kcenter_greedy(X, K):
#     mu = X.mean(dim=0, keepdim=True).cuda()  # 以X的均值初始化
#     indsAll = []
#     D2 = torch.cdist(X, mu).squeeze(1)  # 计算每个点到均值的距离
#     while len(indsAll) < K:
#         ind = D2.argmax().item()  # 选择距离最大的点
#         indsAll.append(ind)
#         mu = torch.cat([mu, X[ind].unsqueeze(0)], dim=0)  # 更新均值
#         D2 = torch.min(D2, torch.cdist(X, mu[-1:].cuda()).squeeze(1))  # 更新D2,只更新最新点的距离

#     return indsAll

# def get_all_outputs(model, unlabeled_loader, use_feature=False):
#     model.eval()
#     outputs = []
#     with torch.no_grad():
#         for inputs, _, _ in unlabeled_loader:
#             inputs = inputs.cuda()
#             out, fea = model(inputs)  # 直接调用模型，避免不必要的变量处理
#             if use_feature:
#                 out = fea
#             else:
#                 out = F.softmax(out, dim=1)
#             outputs.append(out)

#     return torch.cat(outputs, dim=0).cuda()  # 批量化操作，返回完整的张量
