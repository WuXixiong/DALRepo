from .almethod import ALMethod
import torch
import numpy as np

class Uncertainty(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, selection_method="CONF", **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        selection_choices = ["CONF", "Entropy", "Margin", "MeanSTD", "AdversarialBIM"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method
        # for AdversarialBIM
        self.eps = self.args.eps

    def run(self):
        scores = self.rank_uncertainty()
        selection_result = np.argsort(scores)[:self.args.n_query]
        return selection_result, scores

    def rank_uncertainty(self):
        self.models['backbone'].eval()
        selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

        scores = np.array([])
        batch_num = len(selection_loader)
        print("| Calculating uncertainty of Unlabeled set")
        for i, data in enumerate(selection_loader):
            inputs = data[0].to(self.args.device)
            if i % self.args.print_freq == 0:
                print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
            if self.selection_method == "AdversarialBIM":
                self.models['backbone'].train()
                self.models['backbone'] = self.models['backbone'].to(self.args.device)
                scores = np.append(scores, self.cal_dis_bim(inputs).cpu().numpy())
            else:
                with torch.no_grad():
                    if self.selection_method == "CONF":
                        preds, _ = self.models['backbone'](inputs)
                        confs = preds.max(axis=1).values.cpu().numpy()
                        scores = np.append(scores, confs)
                    elif self.selection_method == "Entropy":
                        preds, _ = self.models['backbone'](inputs)
                        preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                        entropys = (np.log(preds + 1e-6) * preds).sum(axis=1)
                        scores = np.append(scores, entropys)
                    elif self.selection_method == 'Margin':
                        preds, _ = self.models['backbone'](inputs)
                        preds = torch.nn.functional.softmax(preds, dim=1)
                        preds_argmax = torch.argmax(preds, dim=1)
                        max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                        preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                        preds_sub_argmax = torch.argmax(preds, dim=1)
                        margins = (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy()
                        scores = np.append(scores, margins)

        if self.selection_method == "MeanSTD":
            probs = self.predict_prob_dropout_split(self.unlabeled_set, selection_loader, n_drop=self.args.n_drop)
            # 将Tensor转移到CPU并转换为NumPy数组
            probs_np = probs.cpu().numpy()
            # 计算标准差
            sigma_c = np.std(probs_np, axis=0)
            # sigma_c = np.std(probs, axis=0)
            uncertainties = np.mean(sigma_c, axis=-1)
            scores = np.append(scores, uncertainties)

        return scores

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores

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
            for i in range(n_drop):

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


    def cal_dis_bim(self, x):
        self.models['backbone'].train()
        self.models['backbone'] = self.models['backbone'].to(self.args.device)

        nx = x.to(self.args.device)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape, device=self.args.device)  # Automatically set device during creation


        # First prediction to get the initial class labels
        out, _ = self.models['backbone'](nx)
        initial_pred = out.max(1)[1].detach()

        while True:
            out, _ = self.models['backbone'](nx + eta)
            current_pred = out.max(1)[1]

            if not torch.all(current_pred == initial_pred):
                break

            loss = torch.nn.functional.cross_entropy(out, initial_pred)
            # loss.requires_grad = True
            loss.backward()

            # eta.data += self.eps * torch.sign(nx.grad.data)
            # nx.grad.data.zero_()
            eta.data += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()

        return (eta * eta).sum()
