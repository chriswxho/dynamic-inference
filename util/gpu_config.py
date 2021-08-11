import torch

def get_batch_size():
    name = torch.cuda.get_device_name(0)
    
    if '2080' in name:
        return 1
    elif '3090' in name or \
        ('TITAN' in name and 'RTX' in name) or \
        'V100' in name:
        return 4
    elif 'A40' in name:
        return 8
    else:
        return 1