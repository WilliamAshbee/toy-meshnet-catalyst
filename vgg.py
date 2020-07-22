import torch.nn as nn
import math
import torch
__all__ = [
    'VGG', 'vgg13',
]

x_encoding = torch.ones((32,32)).float().cuda()
y_encoding = torch.ones((32,32)).float().cuda()


for i in range(32):
    x_encoding[:,i-1] = (i)*2.0/31.0 - 1 
    y_encoding[i-1,:] = (31-i)*2.0/31 - 1  


#print(x_encoding[:,0])
#print(y_encoding[0,:])

x_encoding = x_encoding.unsqueeze(0).unsqueeze(0)
y_encoding = y_encoding.unsqueeze(0).unsqueeze(0)


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000+2):#predict x,y n rs
        super(VGG, self).__init__()
        self.results =  None
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()


    def forward(self, x):
        assert x_encoding.shape == (1,1,32,32)
        assert y_encoding.shape == (1,1,32,32)
        
        x = torch.cat([x, x_encoding.repeat(x.shape[0],1,1,1)], 1)
        x = torch.cat([x, y_encoding.repeat(x.shape[0],1,1,1)], 1)
        features = self.features(x)
        x = features
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

    def freeze_layer(self):
        for param in self.features.parameters():
            param.requires_grad = False


def make_layers(cfg):
    layers = []
    in_channels = 5
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model
