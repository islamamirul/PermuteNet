import json
import torch
import numpy as np

ckpt_path = 'checkpoints/zero_padding_best_deeplabv3_resnet50_cityscapes_os16.pth'

model = torch.load(ckpt_path, map_location=torch.device('cpu'))["model_state"]
print('checkpoint Loaded')

model_list = list(model)

for k in range(0, len(model_list)):
        if 'conv' in model_list[k]:
            if not 'convs' in model_list[k]:
               model[model_list[k]] = torch.from_numpy(np.flip(model[model_list[k]].numpy(), 3).copy())

torch.save(model, 'checkpoints/flipped_zero_padding_best_deeplabv3_resnet50_cityscapes_os16.pth')
