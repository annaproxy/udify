import torch
print(torch.cuda.is_available())
hi = torch.device("cuda:0")

test = torch.zeros(10).to(hi)