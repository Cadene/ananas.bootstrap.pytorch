import torch.nn as nn

class MultiCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MultiCrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, net_out, batch):
        out = {}
        out['loss'] = 0
        for i, x in enumerate(net_out):
            out['loss_head,{}'.format(i)] = self.loss(x, batch['class_id'].view(-1))
            out['loss'] += out['loss_head,{}'.format(i)]
        return out