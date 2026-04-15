import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchattacks
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- 1. Dataset Class: CIFAR-100-C & Adversarial Samples ---

class CIFAR100CDataset(torch.utils.data.Dataset):
    """
    Dataloader for CIFAR-100-C benchmarks and generated adversarial artifacts.
    Supports both .npy file loading and directory-based image loading.
    """
    def __init__(self, root, corruption_type=None, level=None, transform=None, is_adv=False):
        self.transform = transform
        self.is_adv = is_adv
        self.root = root

        if is_adv:
            self.data = []
            self.targets = []
            self.classes = [str(i) for i in range(100)]
            self.class_to_idx = {cls: int(cls) for cls in self.classes}

            for class_idx in range(100):
                class_dir = os.path.join(root, f"class_{class_idx}")
                if not os.path.exists(class_dir):
                    continue
                
                # Ensure deterministic indexing via lexicographical sorting
                image_names = sorted(os.listdir(class_dir))
                
                for img_name in image_names:
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        try:
                            img = Image.open(img_path).convert('RGB')
                            self.data.append(img)
                            self.targets.append(class_idx)
                        except Exception as e:
                            print(f"Warning: Failed to load {img_path}: {e}")

        # Mode 2: Loading Official CIFAR-100-C (.npy format)
        else:
            
            if corruption_type is None or corruption_type.lower() in ['clean', 'none']:
                data_npy = os.path.join(root, "clean.npy")
                if not os.path.exists(data_npy):
                    data_npy = os.path.join(root, "test.npy")
            else:
                data_npy = os.path.join(root, f"{corruption_type}.npy")
            
            labels_npy = os.path.join(root, "labels.npy")
            if not os.path.exists(labels_npy):
                labels_npy = os.path.join(root, "test_labels.npy")

            if not os.path.exists(data_npy):
                raise FileNotFoundError(f"Data file not found: {data_npy}")
            if not os.path.exists(labels_npy):
                raise FileNotFoundError(f"Label file not found: {labels_npy}")


            self.data = np.load(data_npy).astype(np.uint8)
            self.targets = np.load(labels_npy).astype(int)

            # Slice dataset based on corruption severity level (10,000 samples per level)
            if level is not None and corruption_type is not None and corruption_type.lower() not in ['clean', 'none']:
                start_idx = (level - 1) * 10000
                end_idx = level * 10000
                if start_idx < len(self.data):
                    self.data = self.data[start_idx:min(end_idx, len(self.data))]
                    self.targets = self.targets[start_idx:min(end_idx, len(self.targets))]
                else:
                    raise ValueError(f"Requested corruption level {level} is out of bounds.")


            self.data = [Image.fromarray(img) for img in self.data]
            self.classes = [str(i) for i in range(100)]
            self.class_to_idx = {cls: int(cls) for cls in self.classes}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# --- 2. Architecture: ResNet-26 Definition ---


class BasicBlock(nn.Module):
    """Standard residual block with two 3x3 convolutions and skip connection."""
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
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
    ResNet-26 architecture optimized for 32x32 image inputs.
    Structure: Conv1 -> 4 layers (3 blocks each) -> AvgPool -> FC.
    """
    def __init__(self, num_classes=100):
        super(ResNet26_CIFAR, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 3, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
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
    
# --- 3. Adversarial Generation: PGD Pipeline ---

def generate_pgd_cifar100c():
    """
    Main pipeline for generating adversarial perturbations on corrupted domains.
    1. Initializes ResNet-26 backbone with pre-trained weights.
    2. Integrates normalization into the computational graph.
    3. Executes Projected Gradient Descent (PGD) attack.
    4. Persists adversarial samples to disk for robust evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        'root': './data/CIFAR-100-C', 
        'output_root': './results/CIFAR-100-C-Adv',
        'corruption_type': 'gaussian_noise',
        'level': 3,
        'batch_size': 100,
        'pretrained_path': '../CIFAR100-c/resnet26_cifar100_finetuned.pth' 
    }

    output_dir = os.path.join(config['output_root'], f"{config['corruption_type']}-level{config['level']}-PGD-adv")

    # Initialization: Backbone setup and weight loading
    model_backbone = ResNet26_CIFAR(num_classes=100)
    if os.path.exists(config['pretrained_path']):
        checkpoint = torch.load(config['pretrained_path'], map_location='cpu')
        state_dict = checkpoint['state_dict'] if (isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_backbone.load_state_dict(new_state_dict, strict=True)
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {config['pretrained_path']}")

    # End-to-end wrapper: Integrates data normalization into the model graph
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1))
            self.register_buffer('std', torch.tensor(std).view(-1, 1, 1))
        def forward(self, img):
            return (img - self.mean) / self.std

    model = nn.Sequential(Normalization([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]), model_backbone).to(device).eval()

    # Attack Configuration: Projected Gradient Descent (L-inf constraint)
    atk = torchattacks.PGD(model, eps=0.4, alpha=2/255, steps=20)

    # Data Loader initialization for corrupted distribution
    dataset = CIFAR100CDataset(root=config['root'], corruption_type=config['corruption_type'], level=config['level'], transform=transforms.ToTensor(), is_adv=False)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Full batch generation loop
    print(f"Generating adversarial artifacts to: {output_dir}")
    global_idx = 0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        adv_images = atk(images, labels)

        for i in range(adv_images.size(0)):
            label_idx = labels[i].item()
            class_dir = os.path.join(output_dir, f"class_{label_idx}")
            os.makedirs(class_dir, exist_ok=True)
            save_path = os.path.join(class_dir, f"idx_{global_idx:05d}.png")
            transforms.ToPILImage()(adv_images[i].cpu()).save(save_path)
            global_idx += 1

if __name__ == '__main__':
    generate_pgd_cifar100c()