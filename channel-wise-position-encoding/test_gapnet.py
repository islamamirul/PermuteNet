import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import os

from models import *
from utils import progress_bar


cudnn.benchmark = True
manual_seed = 627937
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
random.seed(manual_seed)


USE_MULTI_GPU = 0
cuda = torch.cuda.is_available()
gpu_id = 0
multi_gpu = True if USE_MULTI_GPU == 1 else False


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
grid_size = 3
batch_size = 48
lr = 0.001
beta1 = 0.5
resume = True

checkpoint_name = 'resnet18_location_GAP_zero_gridsize_' + str(grid_size)


# Data
print('==> Grid Size: ', grid_size)
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
net = ResNet18GAPNet(num_class=grid_size*grid_size)  # Location classification
# net = ResNet18GAPNet(num_class=10)  # Object classification

criterion = nn.CrossEntropyLoss()

# Loading pretrained weights
if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
    checkpoint2 = torch.load('./checkpoints/' + checkpoint_name + '.pth', map_location={'cuda:0': 'cuda:0'})
    checkpoint = checkpoint2['net']
    print("The accuracy of best sanpshot: ", checkpoint2['acc'])
    if USE_MULTI_GPU:
        net.load_state_dict(checkpoint)
    else:
        for k, v in list(checkpoint.items()):
            if k[0:6] == 'module':
                kk = k[7:]
                del checkpoint[k]
                checkpoint[kk] = v
        net.load_state_dict(checkpoint)
    best_acc = checkpoint2['acc']
    start_epoch = checkpoint2['epoch']


if cuda:
    if not multi_gpu:
        net.cuda(gpu_id)
        criterion.cuda(gpu_id)
    else:
        net.cuda()
        net = nn.DataParallel(net)
        criterion.cuda()
        cudnn.benchmark = True


optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))


# Test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct_loc = 0
    total_loc = 0
    with torch.no_grad():
        for batch_idx, (inputs_cifar, targets) in enumerate(testloader):

            # Generating new data for the training
            inputs = torch.zeros((inputs_cifar.shape[0], inputs_cifar.shape[1], inputs_cifar.shape[2] * grid_size,
                                  inputs_cifar.shape[3] * grid_size), dtype=torch.float)

            location_target = torch.zeros((targets.shape[0]), dtype=torch.long)

            for i in range(0, len(inputs_cifar)):
                location = random.randint(1, grid_size * grid_size)
                grid_y_location = np.mod((location - 1), grid_size)
                grid_x_location = int(np.floor((location - 1) / grid_size))

                grid_x1, grid_x2 = grid_x_location * inputs_cifar.shape[2], grid_x_location * inputs_cifar.shape[2] + \
                                   inputs_cifar.shape[2]
                grid_y1, grid_y2 = grid_y_location * inputs_cifar.shape[2], grid_y_location * inputs_cifar.shape[2] + \
                                   inputs_cifar.shape[2]

                inputs[i, :, grid_x1:grid_x2, grid_y1:grid_y2] = inputs_cifar[i]
                location_target[i] = location - 1

            inputs = inputs.cuda(gpu_id) if not multi_gpu else inputs.cuda()
            location_target = location_target.cuda(gpu_id) if not multi_gpu else location_target.cuda()

            outputs_loc = net(inputs)
            loss = criterion(outputs_loc, location_target)

            test_loss += loss.item()

            # Location Accuracy
            _, predicted_loc = outputs_loc.max(1)
            total_loc += location_target.size(0)
            correct_loc += predicted_loc.eq(location_target).sum().item()

    acc = 100. * correct_loc / total_loc
    print("Accuracy at Location: ", epoch + 1)
    print(acc)

    return acc


mean_acc = []
for epoch in range(0, grid_size*grid_size):
    accuracy = test(epoch)
    mean_acc.append(accuracy)
    np.savetxt('./results_gapnet/' + checkpoint_name + '.txt', mean_acc)
print("Top Location iou: ", np.mean(mean_acc))
