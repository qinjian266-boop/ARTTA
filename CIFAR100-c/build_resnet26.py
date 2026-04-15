"""
ResNet26 for CIFAR-100 with Weights Transferred from ImageNet-Pretrained ResNet34.

Description:
    This module implements a ResNet26 architecture optimized for CIFAR datasets.
    It provides a utility to initialize weights by mapping layers from a 
    pretrained ResNet34 (ImageNet), bypassing the standard ImageNet stem 
    to accommodate 32x32 input resolutions.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# Model Components
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolution
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Identity mapping / Downsampling shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet26_CIFAR(nn.Module):
    """
    ResNet26 Architecture tailored for CIFAR-10/100.
    
    Structure: [3, 3, 3, 3] blocks for Layer 1-4 respectively.
    Note: Initial 7x7 conv and maxpool from standard ResNet are replaced with 
    a 3x3 conv to preserve spatial resolution for small 32x32 images.
    """
    def __init__(self, num_classes=100):
        super(ResNet26_CIFAR, self).__init__()
        self.in_planes = 64
        block = BasicBlock
        num_blocks = [3, 3, 3, 3]  # ResNet26

        # CIFAR optimized head: Smaller kernel and no maxpooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual Layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ==========================================
# Weight Transfer Utilities
# ==========================================
def get_pretrained_resnet26(num_classes=100):
    """
    Initializes a ResNet26 model with weights from an ImageNet-pretrained ResNet34.
    
    Args:
        num_classes (int): Number of output classes for the FC layer.
        
    Returns:
        nn.Module: ResNet26 model with partial pretrained weights.
    """
    print("Initializing ResNet26 (CIFAR)...")
    target_model = ResNet26_CIFAR(num_classes=num_classes)
    
    print("Loading source ResNet34 weights (ImageNet)...")
    source_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    
    print("Transferring layer weights...")
    
    layer_map = {
        'layer1': 3, 
        'layer2': 3, 
        'layer3': 3, 
        'layer4': 3   
    }
    
    transferred_keys = []
    
    for layer_name, num_blocks in layer_map.items():
        src_layer = getattr(source_model, layer_name)
        dst_layer = getattr(target_model, layer_name)
        
        for i in range(num_blocks):
            
            src_block_state = src_layer[i].state_dict()
            
            
            new_state_dict = {}
            for k, v in src_block_state.items():
                if k.startswith("downsample"):
                    new_key = k.replace("downsample", "shortcut")
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            
            dst_layer[i].load_state_dict(new_state_dict)
            transferred_keys.append(f"{layer_name}.{i}")
            
    print(f"Successfully transferred weights for: {list(layer_map.keys())}")
    print("Note: 'conv1' and 'fc' layers are randomly initialized.")
    
    return target_model

if __name__ == "__main__":
    model = get_pretrained_resnet26(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Output shape:  {y.shape}")