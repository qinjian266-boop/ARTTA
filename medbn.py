"""
Robust Median-Based Batch Normalization (MedBN)

Notes:
- Keeps original API names for compatibility (MedBN, find_and_wrap_bns, adapt_model, convert_model_to_medbn, MedBNAdapter)
- Ensures proper parameter and buffer registration
- Uses median-based statistics for improved robustness
"""
import torch
import torch.nn as nn

class MedBN(nn.Module):
    """
    Median-based Batch Normalization layer.

    Key characteristics:
    - Uses channel-wise median instead of mean
    - Variance computed as mean squared deviation from median
    - Blends batch statistics with running statistics using a prior weight
    - Supports affine transformation
    """
    def __init__(self, orig_bn: nn.BatchNorm2d, prior: float = 0.1):
        super().__init__()
 
        self.num_features = orig_bn.num_features
        self.eps = float(getattr(orig_bn, "eps", 1e-5))
        self.affine = bool(getattr(orig_bn, "affine", True))
        self.track_running_stats = bool(getattr(orig_bn, "track_running_stats", True))
        self.prior = float(prior)


        if self.affine:
            init_w = orig_bn.weight.detach().clone() if orig_bn.weight is not None else torch.ones(self.num_features)
            init_b = orig_bn.bias.detach().clone()   if orig_bn.bias  is not None else torch.zeros(self.num_features)
            self.weight = nn.Parameter(init_w)
            self.bias = nn.Parameter(init_b)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            rm = orig_bn.running_mean.detach().clone() if hasattr(orig_bn, 'running_mean') else torch.zeros(self.num_features)
            rv = orig_bn.running_var.detach().clone()  if hasattr(orig_bn, 'running_var')  else torch.ones(self.num_features)
            self.register_buffer('running_mean', rm)
            self.register_buffer('running_var', rv)
            if hasattr(orig_bn, 'num_batches_tracked'):
                self.register_buffer('num_batches_tracked', orig_bn.num_batches_tracked.detach().clone())
        else:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            if hasattr(orig_bn, 'num_batches_tracked'):
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    @staticmethod
    def find_and_wrap_bns(model, prior):
        """
         Identify all BatchNorm2d layers and create corresponding MedBN replacements.

        Returns:
            List of (parent_module, child_name, replacement_module)
        """
        replace_mods = []
        if model is None:
            return []

        for name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                module = MedBN(child, prior)
                replace_mods.append((model, name, module))
            else:
                replace_mods.extend(MedBN.find_and_wrap_bns(child, prior))
        return replace_mods

    @staticmethod
    def adapt_model(model, prior):
        """
        Replace all BatchNorm2d layers in the model with MedBN (in-place).
        """
        replace_mods = MedBN.find_and_wrap_bns(model, prior)
        print(f"| MedBN: Found {len(replace_mods)} BatchNorm2d layers.")
        for parent, name, child in replace_mods:
            setattr(parent, name, child)
        return model

    def forward(self, x):
        """
        Forward pass using median-based normalization.

        Steps:
        1. Compute channel-wise median
        2. Compute variance as mean squared deviation from median
        3. Blend with running statistics
        4. Normalize and apply affine transformation
        """

        batch_median = self._get_median(x).detach()               
        batch_var = self._get_med_var(x, batch_median).detach()   

        
        adaptive_mean = (1.0 - self.prior) * batch_median + self.prior * self.running_mean
        adaptive_var  = (1.0 - self.prior) * batch_var    + self.prior * self.running_var


        x_normalized = (x - adaptive_mean.view(1, -1, 1, 1)) / torch.sqrt(adaptive_var.view(1, -1, 1, 1) + self.eps)
        if self.affine:
            x_normalized = x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x_normalized

    def _get_median(self, x):
        """Compute channel-wise median over (B, H, W)."""
        B, C, H, W = x.shape
        flat = x.permute(1, 0, 2, 3).contiguous().view(C, -1)  # [C, B*H*W]
        med = torch.median(flat, dim=1).values
        return med

    def _get_med_var(self, x, median):
        """Compute variance as mean squared deviation from median."""
        err = x - median.view(1, -1, 1, 1)
        return err.pow(2).mean(dim=(0, 2, 3))  # reduce over B,H,W -> [C]


def convert_model_to_medbn(model, momentum=0.1, eps=1e-5):
    """
    Replace all BatchNorm2d layers with MedBN (in-place).

    Note:
        'momentum' is used as the prior weight for running statistics.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.BatchNorm2d):
            medbn = MedBN(module, prior=momentum)
            setattr(model, name, medbn)
        else:
            convert_model_to_medbn(module, momentum, eps)
    return model


class MedBNAdapter(nn.Module):
    """
    Wrapper module that converts a model to use MedBN during inference or adaptation.
    """
    def __init__(self, model, momentum=0.1, eps=1e-5):
        super(MedBNAdapter, self).__init__()
        self.model = convert_model_to_medbn(model, momentum, eps)
        

    def forward(self, x):
        return self.model(x)

    def cuda(self, device=None):
        self.model = self.model.cuda(device)
        return self

    def cpu(self):
        self.model = self.model.cpu()
        return self

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self
