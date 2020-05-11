import torch
print(torch.cuda.is_available(), torch.device("cuda:0"))