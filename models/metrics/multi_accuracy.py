import torch.nn as nn

from bootstrap.models.metrics.accuracy import accuracy

class MultiAccuracy(nn.Module):

    def __init__(self, topk=[1,5]):
        super(MultiAccuracy, self).__init__()
        self.topk = topk

    def __call__(self, cri_out, net_out, batch):
        out = {}
        for j, x in enumerate(net_out):
            acc_out = accuracy(x.data.cpu(),
                               batch['class_id'].data.cpu(),
                               topk=self.topk)
            for i, k in enumerate(self.topk):
                out['accuracy_top{}_head,{}'.format(k,j)] = acc_out[i]
        return out