import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        if str(torch.__version__[:1])=='1':
            self.nll_loss=nn.NLLLoss(weight,ignore_index=ignore_index, reduction='mean')
        elif float(torch.__version__[:3])>=0.4 :
            self.nll_loss=nn.NLLLoss(weight,size_averge,ignore_index)
        else: #lower than 0.4 or other
            self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)
 
    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0
    
    tqdm_iter = tqdm(
        dataloader,
        total=len(dataloader), 
        leave=False,
        dynamic_ncols=True,
    )

    for _, label, _ in tqdm_iter:
        print("Processing Images...")
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    print("Class Count: ", class_count)
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights
