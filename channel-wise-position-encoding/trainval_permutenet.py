import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random

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
net = ResNet18PermuteNet(num_class=grid_size*grid_size)  # Location classification
# net = ResNet18PermuteNet(num_class=10)  # Object classification

criterion = nn.CrossEntropyLoss()

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


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct_loc = 0
    total_loc = 0
    for batch_idx, (inputs_cifar, targets) in enumerate(trainloader):

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

        optimizer.zero_grad()

        outputs_loc = net(inputs, shuffle=True, layer=5)

        loss = criterion(outputs_loc, location_target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Location Accuracy
        _, predicted_loc = outputs_loc.max(1)
        total_loc += location_target.size(0)
        correct_loc += predicted_loc.eq(location_target).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Loc_Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct_loc / total_loc, correct_loc, total_loc))


# Validation
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

            outputs_loc = net(inputs, shuffle=True, layer=5)
            loss = criterion(outputs_loc, location_target)

            test_loss += loss.item()

            # Location Accuracy
            _, predicted_loc = outputs_loc.max(1)
            total_loc += location_target.size(0)
            correct_loc += predicted_loc.eq(location_target).sum().item()

            acc = 100. * correct_loc / total_loc
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f| Loc_Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), acc, correct_loc, total_loc))

    # Save checkpoint.
    acc = 100. * correct_loc / total_loc
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        torch.save(state, './checkpoints/resnet18_location_GAP_shuffle_zero_gridsize_' + str(grid_size) + '.pth')

        best_acc = acc

    return acc


mean_accuracy = []
for epoch in range(start_epoch, start_epoch+20):
    train(epoch)
    accuracy = test(epoch)
    mean_accuracy.append(accuracy)

print("Top Location iou: ", np.max(mean_accuracy))
