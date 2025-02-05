import torch.nn as nn # type: ignore
from torchvision import models # type: ignore
from torchvision.models import ResNet34_Weights
from MobiCrowdBackend.UNIQUE.BCNN import BCNN

class BaseCNN(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        if self.config.backbone == 'resnet18':
            self.backbone = models.resnet18(weights=ResNet34_Weights.DEFAULT)
        elif self.config.backbone == 'resnet34':
            self.backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        if config.std_modeling:
            outdim = 2
        else:
            outdim = 1
        if config.representation == 'BCNN':
            assert ((self.config.backbone == 'resnet18') | (self.config.backbone == 'resnet34')), "The backbone network must be resnet18 or resnet34"
            self.representation = BCNN()
            self.fc = nn.Linear(512 * 512, outdim)
        else:
            self.fc = nn.Linear(512, outdim)

        if self.config.fc:
            # Freeze all previous layers.
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        """Forward pass of the network."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        if self.config.representation == 'BCNN':
            x = self.representation(x)
        else:
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, start_dim=1)  # type: ignore

        x = self.fc(x)

        if self.config.std_modeling:
            mean = x[:, 0]
            t = x[:, 1]
            var = nn.functional.softplus(t)
            return mean, var
        else:
            return x
