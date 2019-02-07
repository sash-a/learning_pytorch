import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

        self.lin1 = nn.Linear(10, 5)
        self.lin2 = nn.Linear(5, 2)
        self.lin3 = nn.Linear(2, 1)

    def forward(self, inp):
        # inp.requires_grad(True)

        inp = torch.sigmoid(self.lin1(inp))
        inp = torch.sigmoid(self.lin2(inp))
        inp = torch.sigmoid(self.lin3(inp))

        return inp


net = FeedForward()
input = torch.randn(10)
print('in: ', input)

target = torch.randn(1)
print('target: ', target)

output = net(input)
print('out: ', output)

loss_fn = nn.MSELoss()
loss = loss_fn(output, target)
print(loss)

net.zero_grad()
print('grad before back: ', net.lin1.bias.grad)
loss.backward()
print('grad after back:', net.lin1.bias.grad)


lr = 0.1
for param in net.parameters():
    param.data -= param.grad.data * lr
