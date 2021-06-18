import torch
from torch.cuda import device_count

num_gpus = torch.cuda.device_count()
print('number of gpus: ', num_gpus)
for i in range(num_gpus):
    device = torch.cuda.device(i)
    print(torch.cuda.get_device_properties(device))
    print(device)