import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchattacks
from PIL import Image
import os
from tqdm import tqdm

def generate_pgd_attack():
    """
    Generate adversarial examples using PGD for a given dataset.

    For each class, a fixed number of samples are selected and perturbed.
    The adversarial images are saved while preserving the original directory structure.
    """
    # Device configuration (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Input and output directories
    input_dir = './imagenet-C-blur1-200classes'
    output_dir = './imagenet-C-blur1-200classes-PGD-adv'
    print(f"Input directory: {os.path.abspath(input_dir)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")


    # PGD attack parameters
    eps = 0.3      
    alpha = 1/255 
    steps = 100   

    # Load pretrained model (ResNet50)
    pretrained_model = models.resnet50(pretrained=True)

    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 定义一个归一化模块
    class Normalization(nn.Module):
        """Apply channel-wise normalization using ImageNet statistics."""

        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean.to(img.device)) / self.std.to(img.device)

    # Combine normalization layer and pretrained model
    model = nn.Sequential(
        Normalization(mean, std),
        pretrained_model
    ).to(device).eval()

    # Initialize PGD attack
    atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    print("\nPGD attack initialized:")
    print(atk)

    # Define preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load dataset (used for file paths and labels)
    dataset = datasets.ImageFolder(root=input_dir)
    num_classes = len(dataset.classes)
    print(f"\nDataset contains {len(dataset)} images across {num_classes} classes.")

    # Number of samples to generate per class
    num_samples_per_class = 10

    # Track number of selected samples per class
    images_processed_per_class = {class_idx: 0 for class_idx in range(num_classes)}

    # Collect indices of selected samples
    indices_to_process = []

    print("Selecting samples for each class...")
    for i, (path, class_idx) in enumerate(dataset.samples):
        if images_processed_per_class[class_idx] < num_samples_per_class:
            indices_to_process.append(i)
            images_processed_per_class[class_idx] += 1

    total_to_generate = len(indices_to_process)
    print(
        f"Generating up to {num_samples_per_class} samples per class "
        f"({total_to_generate} images in total)."
    )
    print(f"Saving adversarial examples to '{output_dir}'...")

    # Generate and save adversarial examples
    for i in tqdm(indices_to_process, desc="Generating adversarial examples"):
        img_path, class_idx = dataset.samples[i]

        # Load and preprocess image
        original_image_pil = Image.open(img_path).convert('RGB')
        image_tensor = transform(original_image_pil).unsqueeze(0).to(device)
        label = torch.tensor([class_idx]).to(device)

        # Generate adversarial example
        adv_image_tensor = atk(image_tensor, label)

        # Construct output path (preserve directory structure)
        relative_path = os.path.relpath(img_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save adversarial image
        adv_image_pil = transforms.ToPILImage()(adv_image_tensor.squeeze(0).cpu())
        adv_image_pil.save(output_path)

    print(f"\nSuccessfully generated and saved {total_to_generate} adversarial examples.")

if __name__ == '__main__':
    generate_pgd_attack() 