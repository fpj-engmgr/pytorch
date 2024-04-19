import torch
import math

if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, no cuda here.')

if torch.backends.mps.is_available():
    print('I just saw your Apple silicon GPU!')
else:
    print('Nothing to see here')

if torch.backends.mps.is_available():
    gpu_rand = torch.rand(2, 2, device='mps')
    print(gpu_rand)
else:
    print('Sorry, CPU only.')

if torch.cuda.is_available():
    my_device = torch.device('cuda')
elif torch.backends.mps.is_available():
    my_device = torch.device('mps')
else:
    my_device = torch.device('cpu')

print('Device: {}'.format(my_device))
print('We have ', torch)
x = torch.rand(2, 2, device=my_device)
print(x)

