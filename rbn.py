"""
A stateless, robust batch normalization layer (RBN).
This is the correct implementation for the MedBN defense's normalization part.
It uses the median for robust, on-the-fly statistics calculation and does NOT
maintain or update its own running statistics. It relies on a TTA method
like Tent to adapt its affine parameters. This version is self-contained and
requires no external conf.py.
"""
import torch
from torch import nn


class RBN(nn.Module):
    @staticmethod
    def find_bns(parent, prior):
        """Recursively finds all BatchNorm2d layers for replacement."""
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            # Skip any layers that might have already been replaced
            if isinstance(child, RBN):
                continue

            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = RBN(child, prior)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(RBN.find_bns(child, prior))
        return replace_mods

    @staticmethod
    def adapt_model(model, prior):
        """Replaces all BatchNorm2d layers with this RBN layer."""
        replace_mods = RBN.find_bns(model, prior)
        print(f"| RBN (Stateless): Found and replaced {len(replace_mods)} BatchNorm2d layers.")
        for parent, name, child in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior):
        super().__init__()
        self.layer = layer
        self.layer.eval()

        # Inherit affine parameters (weight, bias) which will be adapted by Tent/EATA
        self.weight = layer.weight
        self.bias = layer.bias

        # Inherit other properties
        self.eps = layer.eps
        self.num_features = layer.num_features
        self.affine = layer.affine
        self.prior = prior

        # Hardcode to use median for robust statistics, removing dependency on conf.py
        self.bn_stat = "median"

    def forward(self, input):
        """
        Performs stateless normalization. It does NOT use self.training and
        does NOT update any internal running statistics.
        """
        # Calculate batch statistics on-the-fly
        b_mean = self.find_median(input)
        b_var = self.find_med_var(input, b_mean)

        # Mix batch statistics with the model's pre-trained statistics
        mean = (1 - self.prior) * b_mean + self.prior * self.layer.running_mean
        var = (1 - self.prior) * b_var + self.prior * self.layer.running_var

        # Perform the normalization
        input = (input - mean.view(1, -1, 1, 1)) / (
            torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        )
        if self.affine:
            input = (
                    input * self.weight.view(1, -1, 1, 1)
                    + self.bias.view(1, -1, 1, 1)
            )
        return input

    def find_median(self, input_data):
        shape = input_data.shape
        input_reshaped = input_data.transpose(1, 0).reshape(shape[1], -1)
        median = input_reshaped.median(1)[0]
        return median


    def find_med_var(self, input, median):
        """Calculates variance based on the median for each channel."""
        err = input - median.view(1, -1, 1, 1)
        # Calculate the mean of squared errors across batch and spatial dimensions
        var = err.pow(2).mean(dim=[0, 2, 3])
        return var