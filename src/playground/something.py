import torch

x = torch.randn([3, 5], requires_grad=True)
y = torch.randn([3, 5], requires_grad=True)
z = x * y

print(z.requires_grad, z.grad)

print(y.grad, x.grad)
z.sum().backward()
print(y.grad, x.grad, sep='\n\n')

# x.zero_grad()
# y.zero_grad()
#
# print(y.grad, x.grad, sep='\n\n')
