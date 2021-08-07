import torch

def get_batch_size():
    name = torch.cuda.get_device_name(0)
    
    if name == 'GeForce RTX 2080 Ti':
        return 1
    elif name == 'GeForce RTX 3090':
        return 2
    elif name == 'Tesla V100-SXM2-32GB':
        return 4
    # a40 stats
    else:
        return 1