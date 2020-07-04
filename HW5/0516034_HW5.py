# coding: utf-8

import numpy as np
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score

VGG11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
VGG13 = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


class Net(nn.Module):
    def __init__(self, num_classes, archs):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.extract_feature = self.make_layers(archs)
        self.classifier = nn.Linear(in_features=512, out_features=self.num_classes)

    def make_layers(self, archs):
        layers = []
        in_channels = 3
        for arch in archs:
            if arch == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=arch, kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(arch))
                layers.append(nn.ReLU())
                in_channels = arch
        return nn.Sequential(*layers)

    def forward(self, x):
        feature = self.extract_feature(x).view(x.size(0), -1)
        result = self.classifier(feature)
        return result


class FromNPY(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = torch.from_numpy(target).view(-1).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    WORKING_DIR = './Drive/Colab_intro_to_PR_HW5'

    x_train = np.load(f'{WORKING_DIR}/x_train.npy')
    y_train = np.load(f'{WORKING_DIR}/y_train.npy')
    x_test = np.load(f'{WORKING_DIR}/x_test.npy')
    y_test = np.load(f'{WORKING_DIR}/y_test.npy')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # It's a multi-class classification problem
    class_index = {
        'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
        'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
    }
    print(np.unique(y_train))

    # x_train_flip = np.array(x_train).reshape((50000, 32, 32, 3))
    # x_train_flip = x_train_flip[:, :, ::-1, :]
    # x_train = np.vstack((x_train, x_train_flip))
    # y_train = np.vstack((y_train, y_train))

    # ![image](https://img-blog.csdnimg.cn/20190623084800880.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lqcDE5ODcxMDEz,size_16,color_FFFFFF,t_70)

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.Resize(224),
            torchvision.transforms.ColorJitter(0.1, 0.3, 0.3, 0.3),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor()
        ]
    )
    test_transforms = torchvision.transforms.Compose(
        [
            # torchvision.transforms.ToPILImage(),
            # torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor()
        ]
    )
    train_dataset = FromNPY(x_train, y_train, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = FromNPY(x_test, y_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    TRAIN = True
    TEST = True
    EPOCH = 125
    cuda_flat = torch.cuda.is_available()

    if TRAIN:
        import os
        if not os.path.exists(f'{WORKING_DIR}/model'):
            print('create model folder')
            os.mkdir(f'{WORKING_DIR}/model')
        if not os.path.exists(f'{WORKING_DIR}/log'):
            print('create log folder')
            os.mkdir(f'{WORKING_DIR}/log')
        net = Net(10, VGG19)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        if cuda_flat:
            torch.backends.cudnn.benchmark = True
            net.cuda()
            print('use Cuda')

        print(datetime.now())
        for epoch in range(EPOCH):
            if epoch <= 90:
                lr = 0.000001071429 * epoch * epoch - 0.0002035714 * epoch + 0.01
                set_learning_rate(optimizer, lr)
            # if epoch == 20:
            #     set_learning_rate(optimizer, 0.005)
            # elif epoch == 40:
            #     set_learning_rate(optimizer, 0.003)
            # elif epoch == 70:
            #     set_learning_rate(optimizer, 0.001)
            # elif epoch == 90:
            #     set_learning_rate(optimizer, 0.0008)
            # elif epoch == 110:
            #     set_learning_rate(optimizer, 0.0002)
            train_loss = 0.0
            correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                if cuda_flat:
                    inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                if i%100 == 99:
                    log = f'epoch: {epoch}, batch: {i}, accuracy: {(float(correct)/float(total)):.3f}%, loss: {train_loss}'
                    print(log)
                    print(log, file=open(f'{WORKING_DIR}/log/{epoch}.log', 'a+'))
                    train_loss = 0
                    correct = 0
                    total = 0
            print(datetime.now())

            epoch_correct = 0
            epoch_total = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    if cuda_flat:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_total += labels.size(0)
                    epoch_correct += predicted.eq(labels.data).cpu().sum().item()
            log = f'epoch: {epoch}, accuracy on 10000 test images: {(100 * epoch_correct / epoch_total)}%'
            print(log)
            print(log, file=open(f'{WORKING_DIR}/log/{epoch}.log', 'a+'))
            torch.save(net.state_dict(), f'{WORKING_DIR}/model/net_epoch{epoch}.pth')
        print('Finished Training')
        torch.save(net.state_dict(), f'{WORKING_DIR}/model/net_final.pth')
        print(datetime.now())

    if TEST:
        net = Net(10, VGG19)
        net.load_state_dict(torch.load(f'{WORKING_DIR}/model/net_final.pth'))
        if cuda_flat:
            torch.backends.cudnn.benchmark = True
            net.cuda()
            print('use Cuda')
        correct = 0
        total = 0
        y_pred = None
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                if cuda_flat:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                if y_pred is None:
                    y_pred = np.array(predicted.cpu().numpy())
                else:
                    y_pred = np.append(y_pred, predicted.cpu().numpy())
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum().item()
        print(f'Accuracy of the network on the 10000 test images: {(100 * correct / total)}%')

        # ## DO NOT MODIFY CODE BELOW!
        # please screen shot your results and post it on your report
        assert y_pred.shape == (10000,)
        print(f'\n\nAccuracy of my model on test-set: {accuracy_score(y_test, y_pred)}\n\n')
