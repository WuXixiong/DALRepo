from .almethod import ALMethod
import torch
import numpy as np
from tqdm import tqdm

class EPIG(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    def select(self, **kwargs):
        scores = self.rank_uncertainty()
        selected_indices = np.argsort(scores)[:self.args.n_query]
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores

    def rank_uncertainty(self):
        self.models['backbone'].eval()
        selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

        scores = np.array([])
        batch_num = len(selection_loader)
        print("| Calculating uncertainty of Unlabeled set")

        probs = self.predict_prob_dropout_split(self.unlabeled_set, selection_loader, n_drop=self.args.n_drop)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1
        # 将Tensor转移到CPU并转换为NumPy数组
        uncertainties_np = -uncertainties.cpu().numpy()  # 取负
        scores= np.append(scores, uncertainties_np)

        return scores

    def predict_prob_dropout_split(self, to_predict_dataset, to_predict_dataloader, n_drop):

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.models['backbone'].train()
        self.models['backbone'] = self.models['backbone'].to(self.args.device)

        # Create a tensor to hold probabilities
        probs = torch.zeros([n_drop, len(to_predict_dataset), len(self.args.target_list)]).to(self.args.device)

        # Create a dataloader object to load the dataset
        # to_predict_dataloader = torch.utils.data.DataLoader(to_predict_dataset, batch_size=self.args['batch_size'], shuffle=False)

        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in tqdm(range(n_drop)):

                evaluated_instances = 0
                # for i, data in enumerate(selection_loader):
                # inputs = data[0].to(self.args.device)
                for _, elements_to_predict in enumerate(to_predict_dataloader):
                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict[0].to(self.args.device)
                    out = self.models['backbone'](elements_to_predict)[0]
                    # print(out)
                    pred = torch.nn.functional.softmax(out, dim=1)

                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + elements_to_predict.shape[0]
                    probs[i][start_slice:end_slice] = pred
                    evaluated_instances = end_slice

        return probs

    