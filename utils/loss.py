import torch


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''

    def __init__(self, d=2, p=2, reduction='mean'):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples, -
                                                           1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction == 'mean':
            return torch.mean(all_norms)
        else:
            return torch.sum(all_norms)

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction == 'mean':
            return torch.mean(diff_norms/y_norms)
        else:
            return torch.sum(diff_norms/y_norms)

    def __call__(self, x, y):
        return self.rel(x, y)
