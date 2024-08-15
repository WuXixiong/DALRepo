import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from .almethod import ALMethod


class Uncertainty(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
# def AV_sampling_temperature(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    def select(self, **kwargs):
        Len_labeled_ind_train = len(self.I_index)
        # model.eval()
        with torch.no_grad():
            unlabeledloader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            queryIndex = []
            labelArr = []
            uncertaintyArr = []
            S_ij = {}
            for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
                data, labels = data.to(self.args.device), labels.to(self.args.device) 
                _, outputs = model(data)
                queryIndex += index
                labelArr += list(np.array(labels.cpu().data))
                # activation value based
                v_ij, predicted = outputs.max(1)
                for i in range(len(predicted.data)):
                    tmp_class = np.array(predicted.data.cpu())[i]
                    tmp_index = index[i]
                    tmp_label = np.array(labels.data.cpu())[i]
                    tmp_value = np.array(v_ij.data.cpu())[i]
                    if tmp_class not in S_ij:
                        S_ij[tmp_class] = []
                    S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])


        # fit a two-component GMM for each class
        tmp_data = []
        for tmp_class in S_ij:
            S_ij[tmp_class] = np.array(S_ij[tmp_class])
            activation_value = S_ij[tmp_class][:, 0]
            if len(activation_value) < 2:
                continue
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(np.array(activation_value).reshape(-1, 1))
            prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
            # 得到为known类别的概率
            prob = prob[:, gmm.means_.argmax()]
            # 如果为unknown类别直接为0
            if tmp_class == args.known_class:
                prob = [0]*len(prob)
                prob = np.array(prob)

            if len(tmp_data) == 0:
                tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
            else:
                tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))


        tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
        tmp_data = tmp_data.T
        queryIndex = tmp_data[2][-args.query_batch:].astype(int)
        labelArr = tmp_data[3].astype(int)
        queryLabelArr = tmp_data[3][-args.query_batch:]
        precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
        recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
        return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall