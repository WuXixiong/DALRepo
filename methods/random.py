from .almethod import ALMethod
# import random
import numpy as np

class Random(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    def select(self):
        Q_index = np.random.choice(len(self.U_index), size=self.args.n_query, replace=False)
        scores = list(np.ones(len(self.U_index)))  # equally assign 1 (meaningless)

        return Q_index, scores