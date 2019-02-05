import torch
import torch.nn as nn


class bor(nn.Module):
    def __init__(self):
        super(bor, self).__init__()
        self.lr = 0.9
        self.loss_fn = nn.MSELoss()
        self.h = nn.Linear(2, 1)
        self.h.bias.data.fill_(0.2)  # setting bias

    def forward(self, x):
        return torch.sigmoid(self.h(x))

    @staticmethod
    def percep(t):
        t = t.foreach([1 if x > 0.5 else 0 for x in t])
        return t

    def learn(self, x, y, n=1000):
        for i in range(n):
            for i in range(len(y)):
                inp = x[i]
                exp = y[i]

                out = self.forward(inp)
                # print(out)
                # print(exp)
                loss = self.loss_fn(out, exp)

                self.zero_grad()
                loss.backward()

                print('loss', loss)
                print('in ', inp, '\nout ', out, '\nexpected ', exp)
                print()

                for p in self.parameters():
                    p.data -= self.lr * p.grad.data


inp = [torch.Tensor([0., 0.]),
       torch.Tensor([0., 1.]),
       torch.Tensor([1., 1.]),
       torch.Tensor([1., 1.])]


expected = [torch.Tensor([0.]), torch.Tensor([1.]), torch.Tensor([1.]), torch.Tensor([1.])]

print('in: ', inp)
net = bor()
net.learn(inp, expected, 100)
