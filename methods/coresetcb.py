from .almethod import ALMethod
import torch
import numpy as np
import copy
from tqdm import tqdm

class CoresetCB(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.cur_cycle = cur_cycle
        self.I_index = I_index
        self.labeled_in_set = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)

    def get_features(self):
        self.models['backbone'].eval()
        labeled_features, unlabeled_features = None, None
        with torch.no_grad():
            labeled_in_loader = torch.utils.data.DataLoader(self.labeled_in_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            unlabeled_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

            unlabeled_probs = []
            # generate entire labeled_in features set
            for data in labeled_in_loader:
                inputs = data[0].to(self.args.device)
                out, features = self.models['backbone'](inputs)

                if labeled_features is None:
                    labeled_features = features
                else:
                    labeled_features = torch.cat((labeled_features, features), 0)

            # generate entire unlabeled features set
            for data in unlabeled_loader:
                inputs = data[0].to(self.args.device)
                unlabel_out, features = self.models['backbone'](inputs)
                prob = torch.nn.functional.softmax(unlabel_out, dim=1).cpu().numpy()
                unlabeled_probs.append(prob)
                if unlabeled_features is None:
                    unlabeled_features = features
                else:
                    unlabeled_features = torch.cat((unlabeled_features, features), 0)
            unlabeled_probs = np.vstack(unlabeled_probs) # convert preds to a 2-d nparray
        return unlabeled_probs, labeled_features, unlabeled_features

    def k_center_greedy(self, labeled, unlabeled, n_query, probs):

        if self.args.dataset == 'CIFAR10':
            num_classes = 10
            lamda = 5
        else:
            num_classes = 100
            lamda = 50
        
        labelled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)
        labelled_classes = [labelled_subset[i][1] for i in range(len(labelled_subset))]
        _, counts = np.unique(labelled_classes, return_counts=True)
        class_threshold=int((2*self.args.n_query+(self.cur_cycle+1)*self.args.n_query)/int(self.args.n_class))
        class_share=class_threshold-counts
        samples_share= np.array([0 if c<0 else c for c in class_share]).reshape(int(self.args.n_class),1)

        N = len(probs)
        z=np.zeros(N, dtype=bool)
        probs = np.array(probs)
        Q = copy.deepcopy(probs)

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = torch.min(torch.cdist(labeled[0:2, :], unlabeled), 0).values
        for j in range(2, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist_matrix = torch.cdist(labeled[j:j + 100, :], unlabeled)
            else:
                dist_matrix = torch.cdist(labeled[j:, :], unlabeled)
            min_dist = torch.stack((min_dist, torch.min(dist_matrix, 0).values))
            min_dist = torch.min(min_dist, 0).values
        min_dist = min_dist.reshape((1, min_dist.size(0)))
        farthest = torch.argmax(min_dist)
        # greedy_indices = torch.tensor([farthest])
        greedy_indices = torch.tensor([], dtype=torch.int64)

        # use class-balanced selection
        mat_min = min_dist.squeeze()
        mat_min_values = mat_min.cpu().detach().numpy()
        # mat_min_values = mat_min_values.detach().numpy()
        for i in tqdm(range(n_query)):
            # q_idx_ 是在过滤后的（已经选择了一些query_index后的）未标记样本集中的索引。
            # z_idx 是 q_idx_ 在原始完整未标记样本集中的对应索引。
            # ？？但是这样的话不是每次index都会减小吗？在原本实现中，mat是一个包括未标注和已标注的矩阵，未标注确实每次减小1.
            SAMPLE_SHARE = np.tile(samples_share, N-i) # 表示每个类在剩余未选样本中还需要的样本数量，扩展到与剩余未标注样本数量相同的维度。
            P_Z = np.tile(np.matmul(np.transpose(probs), z), (N-i,1)) # 表示当前已选样本对类别概率的影响。z 是一个布尔数组，表示哪些样本已经被选中，probs 是未标注样本的预测概率矩阵。
            X = SAMPLE_SHARE - np.transpose(Q) - np.transpose(P_Z) # 表示当前类别样本数量与目标类别样本数量的差异。

            q_idx_= np.argmin(-mat_min_values + (lamda/num_classes) * np.linalg.norm(X,axis=0,ord=1)) #兼顾贪心和类平衡
            z_idx = np.arange(N)[~z][q_idx_] # z_idx 是 q_idx_ 在原始完整未标记样本集中的对应索引。
            z[z_idx] = True

            Q = np.delete(Q, z_idx, axis=0)
            mat_min_values = np.delete(mat_min_values, z_idx, axis=0)

            farthest = torch.tensor([z_idx])
            greedy_indices = torch.cat((greedy_indices, farthest), 0) # 加入greedy_indices样本集也就是选定样本

            dist_matrix = torch.cdist(unlabeled[greedy_indices[-1], :].reshape((1, -1)), unlabeled) # 新选中的未标记到其他所有未标记的欧几里得距离
            min_dist = torch.stack((min_dist, dist_matrix)) # 堆叠：迄今为止所有已标记样本与未标记样本之间的最小距离的记录，上述新样本记录
            min_dist = torch.min(min_dist, 0).values # 取最小值，反映当前所有已标记样本（包括刚刚的新样本）与未标记样本之间的最小距离

        return greedy_indices.cpu().numpy()

    def select(self, **kwargs):
        unlabeled_probs, labeled_features, unlabeled_features = self.get_features()
        selected_indices = self.k_center_greedy(labeled_features, unlabeled_features, self.args.n_query, unlabeled_probs)
        scores = list(np.ones(len(selected_indices))) # equally assign 1 (meaningless)

        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores