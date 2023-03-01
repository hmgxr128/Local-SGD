import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



__all__ = ['resnet20_gn', 'resnet32_gn', 'resnet44_gn', 'resnet56_gn', 'resnet110_gn', 'resnet1202_gn']

def _weights_init(m):

    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0, std=0.01)
        m.bias.data.zero_()
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, n_groups=8, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)


        self.bn1 = nn.GroupNorm(n_groups, planes)
        self.bn2 = nn.GroupNorm(n_groups, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':

                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),nn.GroupNorm(n_groups, self.expansion * planes)
                )

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.silu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_groups=8, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.n_groups = n_groups

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(self.n_groups, 16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.n_groups, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20_gn(n_groups):
    return ResNet(BasicBlock, [3, 3, 3], n_groups=n_groups)


def resnet32_gn(n_groups):
    return ResNet(BasicBlock, [5, 5, 5], n_groups=n_groups)


def resnet44_gn(n_groups):
    return ResNet(BasicBlock, [7, 7, 7], n_groups=n_groups)


def resnet56_gn(n_groups):
    return ResNet(BasicBlock, [9, 9, 9], n_groups=n_groups)


def resnet110_gn(n_groups):
    return ResNet(BasicBlock, [18, 18, 18], n_groups=n_groups)


def resnet1202_gn(n_groups):
    return ResNet(BasicBlock, [200, 200, 200], n_groups=n_groups)

