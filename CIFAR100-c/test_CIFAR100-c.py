"""
Experiment: CIFAR100-C classification on ResNet-26.
Description: Evaluates the robustness of ARTTA against Distribution Invading Attack (DIA) 
             during Test-Time Adaptation (TTA).

Main Configuration:
    - Task: CIFAR100-C (Robustness evaluation)
    - Architecture: ResNet-26
    - DIA Attack: 
        - Attack Ratio: 0.2
        - Budget (Epsilon): 0.4 (L_inf norm)
        - Iterations: 20 steps (Alpha: 2/255)
        - Strategy: Untargeted, random start
    - MedBN Settings:
        - Prior (Mixing Coefficient): 0.6
    - Batch Size: 100
"""

"""
This framework is extensively modified and extended from the original SAR implementation to support adversarial robustness evaluation in TTA.
"""

import sys
import os
# Project environment setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import time
import random
import math
import numpy as np
from collections import defaultdict
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import argparse  
# Method-specific imports
import tent
import eata
import sar
from sam import SAM
from utils.utils import get_logger
from utils.cli_utils import AverageMeter, ProgressMeter, accuracy
import adv_filter
from PIL import Image  
from copy import deepcopy
import sotta_utils.sotta as sotta_module
import copy
from typing import List, Optional, Tuple
import logging

# =====================================================================
# GLOBAL CONFIGURATION
# =====================================================================
CONFIG = {
    # Dataset Paths (CIFAR-100-C Standard)
    'cifar100c_root': './data/CIFAR-100-C',  
    'corruption_type': 'gaussian_noise',          
    'corruption_level': 3,             
    'adv_data': './data/CIFAR-100-C-Adv/noise-L3-PGD', 
    'output': './results/cifar100_resnet26',     

    # Evaluation Model & TTA Baselines
    'method': 'sotta',  # Experimental target: {no_adapt, tent, eata, sar, sotta}
    'model': 'resnet26',  

    # Static Poisoning: Mixing pre-generated adversarial samples into the data stream
    'use_adv': False,  
    'adv_ratio': 0.2,  # Proportion of adversarial samples per mini-batch (0~1)

    # Dynamic Threat: Distribution Invading Attack (DIA) configuration
    'dia_attack': {
        'enabled': True,
        'attack_ratio': 0.2,  
        'eps': 0.4,  # L_inf perturbation budget
        'steps': 20,  
        'alpha': 2.0 / 255,  # Optimization step size
        'attack_type': 'indiscriminate',  
        'random_start': True,
        'norm': 'Linf'
    },

    # Defensive Component: Multi-feature Adversarial Detection
    'adv_detection': {
        'enabled': True,
        'window_size': 20,  # Sliding window for temporal statistical thresholding
        'feature_weights': [0.1, 0, 0.1, 0.8],  # Weights for [entropy, avg_grad_magnitude , gradient_direction_entropy, mean_spectrum]
        'threshold_method': 'std',  # Heuristic: Standard Deviation based dynamic threshold
        'quantile': 0.8,  
        'std_factor': 0.9,  # Scalar for sigma-based rejection

        'batch_stats': {
            'print_freq': 1,  
            'save_features': False, 
        },

        'min_clean_samples': 1,  # Minimal reliable samples required for TTA update
    },

    # Sample Rejection: Exclusion of compromised samples from gradients update
    'sample_filtering': {
        'enabled': True,  
    },


    # Execution Environment
    'seed': 2025,
    'gpu': 0,  # GPU ID
    'workers': 0,
    'test_batch_size': 100,
    'print_freq': 1,
    'tta_iterations': 1,  # Update steps per batch for online adaptation

    # SAR Hyperparameters
    'sar_margin_e0': math.log(100) * 0.40,  # Reliable sample selection threshold

    # EATA Parameters
    'fisher_size': 2000,  # Sample count for Fisher Information Matrix estimation
    'fisher_alpha': 1.0,  # Trade-off between entropy and regularization
    'e_margin': math.log(100) * 0.40,  # E_0: Entropy-based reliability filter
    'd_margin': 0.4,  # Epsilon: Diversity-based redundancy filter

    # SoTTA Parameters
    'sotta': {
    'mem_type': 'HUS', # Homogeneous Uniform Sampling
    'mem_capacity': 100,
    'hus_batch_size': 100,
    'esm_rho': 0.05,
    'steps': 1
    },

    # MedBN: Robust Batch Normalization via Median Statistics
    'medbn': {
        'enable': False,  
        'prior': 0.6,  # Hybrid weight for Source-Target distribution alignment
    },
}
# =====================================================================


# ================= CIFAR-100 Dataset Statistics =================
# Standard mean and standard deviation for CIFAR-100 normalization
cifar_mean = [0.5071, 0.4867, 0.4408]
cifar_std = [0.2675, 0.2565, 0.2761]

# Define test transformation pipeline
# Note: Spatial transforms (Resize/CenterCrop) are omitted as ResNet-20 
# is designed for the native 32x32 input resolution of CIFAR.
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])

class DIAAttacker:
    """
    White-box Distribution Invading Attack (DIA) for evaluating Test-Time Adaptation robustness.
    Implements PGD-based adversarial perturbation with full model state preservation.
    """

    def __init__(self, model, config: dict, logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None):
        # unwrap wrapper if needed
        self.model = model.model if hasattr(model, 'model') else model
        if hasattr(self.model, 'model'):
            try:
                self.model = self.model.model
            except Exception:
                pass

        self.logger = logger or logging.getLogger("DIAAttacker")
        attack_cfg = config.get('dia_attack', {})

        self.eps_pixel = float(attack_cfg.get('eps', 8.0 / 255.0))
        self.alpha_pixel = float(attack_cfg.get('alpha', 2.0 / 255.0))
        self.steps = int(attack_cfg.get('steps', 10))
        self.attack_ratio = float(attack_cfg.get('attack_ratio', 0.2))
        self.attack_type = attack_cfg.get('attack_type', 'indiscriminate')
        self.norm = attack_cfg.get('norm', 'Linf')

        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device('cpu')

        # assume ImageNet mean/std by default; override config if needed
        self.norm_mean = torch.tensor([0.5071, 0.4867, 0.4408], device=self.device).view(1, 3, 1, 1)
        self.norm_std = torch.tensor([0.2675, 0.2565, 0.2761], device=self.device).view(1, 3, 1, 1)

        # store eps/alpha in normalized input space for updates/clamping
        self.eps_norm = (self.eps_pixel / self.norm_std).to(self.device)
        self.alpha_norm = (self.alpha_pixel / self.norm_std).to(self.device)

        # input bounds in normalized space
        self.input_lower = ((0.0 - self.norm_mean) / self.norm_std).to(self.device)
        self.input_upper = ((1.0 - self.norm_mean) / self.norm_std).to(self.device)

        # internal attack state
        self.perturb = None               # tensor (num_attack, C, H, W) leaf, requires_grad=True
        self.attack_index: List[int] = [] # indices chosen to attack in current batch (sorted)
        self.benign_index: List[int] = []
        self.benign_cls = None            # tensor of benign labels for loss computation

        # saved persistent model state (state_dict + bn/rbn infos)
        self._saved_state = None
        self._saved_bn = {}
        self._orig_training_mode = None

        # statistics
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'confidence_reduction': 0.0,
            'accuracy_drops': []
        }

        # options
        self._make_dirs = False  
        self.debug = bool(attack_cfg.get('debug', False))
        self.random_start = bool(attack_cfg.get('random_start', False))

        self.logger.info(f"[DIA] init: eps_pixel={self.eps_pixel:.6f}, alpha_pixel={self.alpha_pixel:.6f}, "
                         f"eps_norm_mean={self.eps_norm.mean().item():.6f}, steps={self.steps}, ratio={self.attack_ratio}, norm={self.norm}")

    # ---------------- Logging ----------------
    def _log(self, *args, level='info'):
        if self.logger:
            msg = " ".join(map(str, args))
            if level == 'info':
                self.logger.info(msg)
            elif level == 'warning':
                self.logger.warning(msg)
            elif level == 'error':
                self.logger.error(msg)
            else:
                self.logger.debug(msg)
        else:
            print(*args)

    # ---------------- Save/restore persistent state ----------------
    def _save_model_state_and_bn(self, to_cpu_state: bool = True, shallow_copy: bool = True):
        """
        Save model state and BatchNorm/RBN buffers for full restoration after attack.
        """
        model = self.model
        try:
            if shallow_copy:
                self._saved_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
            else:
                self._saved_state = copy.deepcopy(model.state_dict())
                # move to CPU to save GPU memory if requested
                if to_cpu_state:
                    for k, v in list(self._saved_state.items()):
                        if isinstance(v, torch.Tensor):
                            self._saved_state[k] = v.cpu()
        except Exception:
            # fallback: clone each tensor
            self._saved_state = {}
            for k, v in model.state_dict().items():
                try:
                    self._saved_state[k] = v.clone().detach().cpu()
                except Exception:
                    self._saved_state[k] = v.clone().detach()

        # Save BN / RBN persistent fields (store CPU clones)
        self._saved_bn = {}
        for name, m in model.named_modules():
            # detect RBN-like wrapper by heuristics (has .prior and .layer with running_mean)
            if hasattr(m, 'prior') and hasattr(m, 'layer') and hasattr(m.layer, 'running_mean'):
                try:
                    self._saved_bn[name] = {
                        'type': 'rbn',
                        'prior': float(getattr(m, 'prior', 0.0)),
                        'layer_running_mean': m.layer.running_mean.detach().cpu().clone() if getattr(m.layer, 'running_mean', None) is not None else None,
                        'layer_running_var': m.layer.running_var.detach().cpu().clone() if getattr(m.layer, 'running_var', None) is not None else None,
                        'weight': m.weight.detach().cpu().clone() if getattr(m, 'weight', None) is not None else None,
                        'bias': m.bias.detach().cpu().clone() if getattr(m, 'bias', None) is not None else None,
                        'bn_stat': getattr(m, 'bn_stat', None)
                    }
                except Exception:
                    self._saved_bn[name] = {'type': 'rbn', 'prior': float(getattr(m, 'prior', 0.0))}
                continue

            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                try:
                    self._saved_bn[name] = {
                        'type': 'bn',
                        'track_running_stats': bool(getattr(m, 'track_running_stats', True)),
                        'running_mean': m.running_mean.detach().cpu().clone() if getattr(m, 'running_mean', None) is not None else None,
                        'running_var': m.running_var.detach().cpu().clone() if getattr(m, 'running_var', None) is not None else None,
                        'weight': m.weight.detach().cpu().clone() if getattr(m, 'weight', None) is not None else None,
                        'bias': m.bias.detach().cpu().clone() if getattr(m, 'bias', None) is not None else None
                    }
                except Exception:
                    self._saved_bn[name] = {'type': 'bn'}

        # save training mode
        self._orig_training_mode = model.training

    def _configure_model_for_attack(self, force_rbn_prior_zero: bool = True):
        """
        Prepare model for attack:
          - set model.train() so BN/RBN uses batch stats
          - freeze parameter gradients so only perturb is optimized
          - for RBN set prior=0.0 (if desired) to force batch-stat usage; do NOT replace running buffers here
        """
        model = self.model
        try:
            model.train()
        except Exception:
            pass

        # freeze all weights so no param grads during attack
        model.requires_grad_(False)

        for name, m in model.named_modules():
            # RBN wrapper handling: set prior to 0 to rely on batch stats if present
            if hasattr(m, 'prior') and hasattr(m, 'layer'):
                if force_rbn_prior_zero:
                    try:
                        m.prior = 0.0
                    except Exception:
                        pass
                # ensure affine not updated (we don't want to adapt weight/bias here)
                if getattr(m, 'weight', None) is not None:
                    m.weight.requires_grad_(False)
                if getattr(m, 'bias', None) is not None:
                    m.bias.requires_grad_(False)
                # DO NOT change m.layer.running_mean/var here (no assignment)
                continue

            # standard BatchNorm: use batch stats during attack
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                try:
                    m.track_running_stats = False
                except Exception:
                    pass
                if getattr(m, 'weight', None) is not None:
                    m.weight.requires_grad_(False)
                if getattr(m, 'bias', None) is not None:
                    m.bias.requires_grad_(False)

        self._in_attack_mode = True
        self._log("[DIA] model configured for attack: model.train(), weights frozen, RBN.prior set to 0 (if present)")

    def _restore_model_after_attack(self, verify_restore: bool = False, atol: float = 1e-6):
        """
        Restore persistent saved state. Use copy_ to restore buffers so that module buffers remain Tensors
        (avoid direct assignment m.running_mean = saved_tensor which would point to detached CPU tensor).
        """
        model = self.model

        # 1) restore state_dict (try load_state_dict first)
        if getattr(self, '_saved_state', None) is not None:
            try:
                model.load_state_dict(self._saved_state, strict=False)
            except Exception:
                # fallback: per-key copy
                try:
                    cur_state = model.state_dict()
                    for k, v in self._saved_state.items():
                        if k in cur_state:
                            try:
                                cur_state[k].copy_(v.to(cur_state[k].device))
                            except Exception:
                                pass
                except Exception:
                    pass

        # 2) restore bn/rbn buffers using copy_ into existing buffers (move saved cpu tensors back to device)
        for name, m in model.named_modules():
            info = self._saved_bn.get(name, None)
            if info is None:
                continue
            if info.get('type') == 'rbn':
                try:
                    if 'prior' in info:
                        m.prior = float(info['prior'])
                except Exception:
                    pass
                try:
                    if info.get('layer_running_mean') is not None and hasattr(m.layer, 'running_mean'):
                        m.layer.running_mean.copy_(info['layer_running_mean'].to(m.layer.running_mean.device))
                    if info.get('layer_running_var') is not None and hasattr(m.layer, 'running_var'):
                        m.layer.running_var.copy_(info['layer_running_var'].to(m.layer.running_var.device))
                    if info.get('weight') is not None and getattr(m, 'weight', None) is not None:
                        m.weight.copy_(info['weight'].to(m.weight.device))
                    if info.get('bias') is not None and getattr(m, 'bias', None) is not None:
                        m.bias.copy_(info['bias'].to(m.bias.device))
                except Exception:
                    pass
            elif info.get('type') == 'bn':
                try:
                    if 'track_running_stats' in info:
                        m.track_running_stats = bool(info['track_running_stats'])
                    if info.get('running_mean') is not None and hasattr(m, 'running_mean'):
                        m.running_mean.copy_(info['running_mean'].to(m.running_mean.device))
                    if info.get('running_var') is not None and hasattr(m, 'running_var'):
                        m.running_var.copy_(info['running_var'].to(m.running_var.device))
                    if info.get('weight') is not None and getattr(m, 'weight', None) is not None:
                        m.weight.copy_(info['weight'].to(m.weight.device))
                    if info.get('bias') is not None and getattr(m, 'bias', None) is not None:
                        m.bias.copy_(info['bias'].to(m.bias.device))
                except Exception:
                    pass

        # 3) restore training/eval flag
        if getattr(self, '_orig_training_mode', None) is not None:
            try:
                model.train(self._orig_training_mode)
            except Exception:
                if self._orig_training_mode:
                    model.train()
                else:
                    model.eval()

        model.requires_grad_(True)
        self._in_attack_mode = False

        # optional verification/debug
        if verify_restore and self.debug:
            for name, info in self._saved_bn.items():
                if info.get('type') == 'rbn' and 'prior' in info:
                    try:
                        cur_module = dict(model.named_modules()).get(name, None)
                        if cur_module is None:
                            self._log(f"[DIA verify] module {name} missing after restore", level='warning')
                            continue
                        cur_prior = getattr(cur_module, 'prior', None)
                        if cur_prior is None or abs(float(cur_prior) - float(info['prior'])) > atol:
                            self._log(f"[DIA verify] RBN prior mismatch at {name}: saved={info['prior']}, now={cur_prior}", level='warning')
                    except Exception:
                        pass

        self._log("[DIA] model persistent state restored after attack")

    # ---------------- perturb management ----------------
    def reset_perturb(self, images: List[torch.Tensor], cls: List[int], attack_ratio: Optional[float] = None):
        """
        Initialize perturbation tensor and attack indices with optional random start.
        """
        if attack_ratio is None:
            attack_ratio = self.attack_ratio
        batch_size = len(cls)
        num_attack = int(batch_size * float(attack_ratio))
        
        
        if num_attack <= 0:
            self.attack_index = []
            self.benign_index = list(range(batch_size))
            self.perturb = None
            self.benign_cls = torch.tensor([cls[i] for i in self.benign_index], dtype=torch.long, device=self.device)
            self._log(f"[DIA] reset_perturb: batch={batch_size} num_attack=0")
            return

        
        perm = torch.randperm(batch_size)
        attack_idx = perm[:num_attack].sort().values.tolist()
        benign_idx = perm[num_attack:].sort().values.tolist()

        self.attack_index = attack_idx
        self.benign_index = benign_idx
        self.benign_cls = torch.tensor([cls[i] for i in self.benign_index], dtype=torch.long, device=self.device)

        
        sample = images[0]
        if sample.dim() == 3:
            C, H, W = sample.shape
        elif sample.dim() == 4:
            _, C, H, W = sample.shape
        else:
            raise ValueError("unsupported image tensor shape for reset_perturb")

        
        if self.perturb is None or self.perturb.size(0) != num_attack:
            
            self.perturb = torch.zeros((num_attack, C, H, W), device=self.device, dtype=sample.dtype)
            
            if self.random_start:
                self._log("[DIA] Using random start for PGD initialization.")
                noise = torch.rand_like(self.perturb) * 2 - 1 
                noise = noise * self.eps_norm 
                with torch.no_grad():
                    self.perturb.copy_(noise)
                    torch.maximum(self.perturb, -self.eps_norm, out=self.perturb)
                    torch.minimum(self.perturb, self.eps_norm, out=self.perturb)
            
            
            self.perturb.requires_grad_(True)
        else:
            
            with torch.no_grad():
                if self.random_start:
                    noise = torch.rand_like(self.perturb) * 2 - 1
                    noise = noise * self.eps_norm
                    self.perturb.copy_(noise)
                    torch.maximum(self.perturb, -self.eps_norm, out=self.perturb)
                    torch.minimum(self.perturb, self.eps_norm, out=self.perturb)
                else:
                    self.perturb.zero_()

        
        if len(self.attack_index) > 0:
            self.attack_index_tensor = torch.tensor(self.attack_index, dtype=torch.long, device=self.device)
        else:
            self.attack_index_tensor = None

        self._log(f"[DIA] reset_perturb: batch={batch_size} num_attack={num_attack} random_start={self.random_start}")

    def _make_perturbed_batch(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        images = images.to(self.device)
        if self.perturb is None or len(self.attack_index) == 0:
            return images

        # build full perturb map, vectorized
        perturb_full = torch.zeros_like(images, device=self.device)
        if getattr(self, 'attack_index_tensor', None) is None:
            self.attack_index_tensor = torch.tensor(self.attack_index, dtype=torch.long, device=self.device)
        perturb_full.index_copy_(0, self.attack_index_tensor, self.perturb)

        perturbed = images + perturb_full
        # clamp to input bounds (broadcasting)
        torch.maximum(perturbed, self.input_lower)
        torch.minimum(perturbed, self.input_upper)
        return perturbed

    # ---------------- attack core (whitebox autograd) ----------------
    def tta_attack_train(self, images: torch.Tensor, targets: torch.Tensor, step_idx: int = 0) -> float:
        """
        Single attack iteration using autograd. Returns scalar loss value.
        """
        if self.perturb is None or len(self.attack_index) == 0:
            self._log("[DIA] tta_attack_train: no perturbations -> skip", level='warning')
            return 0.0

        model = self.model

        # Build perturbed batch (batch update, avoiding for-loop)
        perturbed_batch = self._make_perturbed_batch(images)

        # Ensure BN modules use batch stats as expected for attack

        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.train()

        # Forward pass (don't wrap with no_grad, as we need gradients)
        preds = model(perturbed_batch)

        # Compute loss based on attack type
        benign_idx_tensor = torch.tensor(self.benign_index, dtype=torch.long, device=self.device)
        benign_preds = preds[benign_idx_tensor] if len(self.benign_index) > 0 else preds[:1]
        if self.attack_type == 'indiscriminate':
            loss = -F.cross_entropy(benign_preds, self.benign_cls)
        elif self.attack_type == 'targeted':
            loss = -F.cross_entropy(benign_preds, self.benign_cls)
        else:
            loss = -F.cross_entropy(benign_preds, self.benign_cls) if len(self.benign_index) > 0 else torch.tensor(0.0, device=self.device)

        # Ensure perturb requires gradient and is a leaf
        if not self.perturb.requires_grad:
            self.perturb.requires_grad_(True)

        # Compute gradients with respect to perturb
        grads = torch.autograd.grad(loss, [self.perturb], retain_graph=False, create_graph=False, allow_unused=True)
        g = grads[0] if grads is not None else None

        if g is None:
            self._log("[DIA] warning: gradient for perturb is None (no update).", level='warning')
            return float(loss.detach().cpu().item())

        # Update perturb in-place under no_grad
        with torch.no_grad():
            if self.norm == 'Linf':
                step = -self.alpha_norm * g.sign()  # broadcastable (1,C,1,1)
                self.perturb.add_(step)
                # per-channel clamp using out= to keep storage
                torch.maximum(self.perturb, -self.eps_norm, out=self.perturb)
                torch.minimum(self.perturb,  self.eps_norm, out=self.perturb)

            elif self.norm == 'L2':
                # normalize grad per sample
                g_view = g.view(g.size(0), -1)
                g_norm = g_view.norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
                step = - self.alpha_norm * (g / g_norm)
                self.perturb.add_(step)

                # projection in normalized space: compute per-sample eps_l2
                N, C, H, W = self.perturb.shape
                eps_expand = self.eps_norm.expand(N, -1, H, W)  # (N,C,H,W)
                eps_l2_per_sample = eps_expand.contiguous().view(N, -1).norm(p=2, dim=1)  # (N,)

                p_view = self.perturb.view(N, -1)
                p_norm = p_view.norm(p=2, dim=1)

                if (p_norm > eps_l2_per_sample).any():
                    mask = (p_norm > eps_l2_per_sample)
                    # compute scaling factors for masked samples
                    # factor shape: (#masked, 1)
                    factor = (eps_l2_per_sample[mask] / (p_norm[mask] + 1e-12)).view(-1, 1)
                    # scale only masked rows in-place on p_view
                    p_view[mask] = p_view[mask] * factor

                    # write back into self.perturb **using copy_** to preserve storage/leaf
                    self.perturb.copy_(p_view.view_as(self.perturb))
                # else: no change needed (already within L2 ball)

            else:
                step = -self.alpha_norm * g.sign()
                self.perturb.add_(step)
                # clamp symmetric per-channel
                torch.maximum(self.perturb, -self.eps_norm, out=self.perturb)
                torch.minimum(self.perturb,  self.eps_norm, out=self.perturb)

        return float(loss.detach().cpu().item())

    def tta_attack_update(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply final perturb and return adv batch.
        """
        B = images.size(0)
        images = images.to(self.device)
        if self.perturb is None or len(self.attack_index) == 0:
            return images

        adv = images.clone()
        for p_idx, idx in enumerate(self.attack_index):
            p = self.perturb[p_idx].detach()
            tmp = images[idx:idx+1] + p.unsqueeze(0)
            adv[idx:idx+1] = torch.max(torch.min(tmp, self.input_upper), self.input_lower)
        return adv

    # ---------------- validation (unchanged semantics) ----------------
    def validate_attack_effectiveness(self, images, targets, adv_images, is_adv, logger=None, debug_samples: int = 5):
        """
        Evaluate attack effect without modifying persistent state (model should be in eval here).
        """
        if logger is None:
            logger = self.logger

        assert images.shape == adv_images.shape, "images and adv_images must have same shape for validation"

        device = self.device
        images = images.to(device)
        adv_images = adv_images.to(device)
        targets = targets.to(device)
        is_adv = is_adv.to(device)

        with torch.no_grad():
            orig_mode = self.model.training
            try:
                self.model.eval()

                benign_mask = (~is_adv).bool()
                attack_mask = is_adv.bool()

                benign_n = int(benign_mask.sum().item())
                attack_n = int(attack_mask.sum().item())

                if benign_n == 0 and attack_n == 0:
                    return {
                        'benign_accuracy_drop': 0.0,
                        'benign_orig_accuracy': 0.0,
                        'benign_adv_accuracy': 0.0,
                        'attack_success_rate': 0.0,
                        'attack_orig_accuracy': 0.0,
                        'attack_adv_accuracy': 0.0,
                        'attack_samples': 0,
                        'benign_samples': 0,
                        'confidence_reduction': 0.0
                    }

                # benign original
                if benign_n > 0:
                    out_orig_benign = self.model(images[benign_mask])
                    probs_orig = F.softmax(out_orig_benign, dim=1)
                    pred_orig = out_orig_benign.argmax(dim=1)
                    correct_orig = (pred_orig == targets[benign_mask]).float()
                    benign_orig_accuracy = correct_orig.mean().item()
                    conf_orig = probs_orig.max(dim=1)[0]
                else:
                    benign_orig_accuracy = 0.0
                    conf_orig = torch.tensor([], device=device)

                # benign adv
                if benign_n > 0:
                    out_adv_benign = self.model(adv_images[benign_mask])
                    probs_adv = F.softmax(out_adv_benign, dim=1)
                    pred_adv = out_adv_benign.argmax(dim=1)
                    correct_adv = (pred_adv == targets[benign_mask]).float()
                    benign_adv_accuracy = correct_adv.mean().item()
                    conf_adv = probs_adv.max(dim=1)[0]
                else:
                    benign_adv_accuracy = 0.0
                    conf_adv = torch.tensor([], device=device)

                benign_accuracy_drop = float(benign_orig_accuracy - benign_adv_accuracy)

                # attack samples
                if attack_n > 0:
                    out_orig_attack = self.model(images[attack_mask])
                    out_adv_attack = self.model(adv_images[attack_mask])

                    pred_orig_attack = out_orig_attack.argmax(dim=1)
                    pred_adv_attack = out_adv_attack.argmax(dim=1)

                    correct_orig_attack = (pred_orig_attack == targets[attack_mask]).float()
                    correct_adv_attack = (pred_adv_attack == targets[attack_mask]).float()

                    attack_success_mask = (correct_orig_attack.bool() & (~correct_adv_attack.bool()))
                    if correct_orig_attack.numel() > 0:
                        attack_success_rate = float(attack_success_mask.float().mean().item())
                        attack_orig_accuracy = float(correct_orig_attack.mean().item())
                        attack_adv_accuracy = float(correct_adv_attack.mean().item())
                    else:
                        attack_success_rate = 0.0
                        attack_orig_accuracy = 0.0
                        attack_adv_accuracy = 0.0

                    probs_orig_attack = F.softmax(out_orig_attack, dim=1)
                    probs_adv_attack = F.softmax(out_adv_attack, dim=1)
                    idxs = torch.arange(probs_orig_attack.size(0), device=device)
                    try:
                        orig_conf_attack = probs_orig_attack[idxs, targets[attack_mask]]
                        adv_conf_attack = probs_adv_attack[idxs, targets[attack_mask]]
                        confidence_reduction = float((orig_conf_attack - adv_conf_attack).mean().item())
                    except Exception:
                        confidence_reduction = 0.0
                else:
                    attack_success_rate = 0.0
                    attack_orig_accuracy = 0.0
                    attack_adv_accuracy = 0.0
                    confidence_reduction = 0.0

                if logger:
                    logger.info(f"[DIA.validate] benign_samples={benign_n}, attack_samples={attack_n}")
                    logger.info(f"[DIA.validate] benign_orig_acc={benign_orig_accuracy:.6f}, benign_adv_acc={benign_adv_accuracy:.6f}, drop={benign_accuracy_drop:.6f}")
                    logger.info(f"[DIA.validate] attack_orig_acc={attack_orig_accuracy:.6f}, attack_adv_acc={attack_adv_accuracy:.6f}, success_rate={attack_success_rate:.6f}, conf_drop={confidence_reduction:.6f}")

                return {
                    'benign_accuracy_drop': benign_accuracy_drop,
                    'benign_orig_accuracy': float(benign_orig_accuracy),
                    'benign_adv_accuracy': float(benign_adv_accuracy),
                    'attack_success_rate': float(attack_success_rate),
                    'attack_orig_accuracy': float(attack_orig_accuracy),
                    'attack_adv_accuracy': float(attack_adv_accuracy),
                    'attack_samples': int(attack_n),
                    'benign_samples': int(benign_n),
                    'confidence_reduction': float(confidence_reduction)
                }
            finally:
                # restore original training flag only
                try:
                    self.model.train(orig_mode)
                except Exception:
                    if orig_mode:
                        self.model.train()
                    else:
                        self.model.eval()

    # ---------------- driver ----------------
    def generate_dia_attack(self,
                            images: torch.Tensor,
                            targets: torch.Tensor,
                            attack_ratio: Optional[float] = None,
                            logger: Optional[logging.Logger] = None,
                            batch_idx: int = 0,
                            debug_check_restore: bool = False
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full DIA attack pipeline: save state → attack → restore state.
        Returns adversarial images and attack mask.
        """
        logger = logger or self.logger
        B = images.size(0)
        if attack_ratio is None:
            attack_ratio = self.attack_ratio
        num_attack = int(B * attack_ratio)

        if num_attack == 0:
            self._log("[DIA] no attacks needed for this batch")
            return images, torch.zeros(B, dtype=torch.bool, device=self.device)

        self._log(f"[DIA] Starting attack on batch size {B}, num_attack {num_attack}")

        # optional debug: compute pre-attack logits for verify
        pre_logits = None
        if debug_check_restore and self.debug:
            try:
                was_training = self.model.training
                self.model.eval()
                for m in self.model.modules():
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                        m.track_running_stats = True
                with torch.no_grad():
                    pre_logits = self.model(images.to(self.device)).detach().cpu()
                self.model.train(was_training)
            except Exception:
                self._log("[DIA debug] failed to compute pre_logits", level='warning')

        # ---- save persistent state and configure model for attack ----
        try:
            self._save_model_state_and_bn()
            self._configure_model_for_attack()

            # initialize perturb
            img_list = [images[i].detach() for i in range(B)]
            cls_list = [int(targets[i].item()) for i in range(B)]
            self.reset_perturb(img_list, cls_list, attack_ratio=attack_ratio)

            iter_losses = []
            for it in range(self.steps):
                loss_val = self.tta_attack_train(images, targets, it)
                iter_losses.append(loss_val)

                # light monitoring
                with torch.no_grad():
                    pert_batch = self._make_perturbed_batch(images)
                    out = self.model(pert_batch)
                    if len(self.benign_index) > 0:
                        benign_idx = torch.tensor(self.benign_index, dtype=torch.long, device=self.device)
                        benign_out = out[benign_idx]
                        probs = F.softmax(benign_out, dim=1)
                        avg_conf = probs.max(dim=1)[0].mean().item()
                        pred = benign_out.argmax(dim=1)
                        acc = (pred == self.benign_cls).float().mean().item()
                    else:
                        avg_conf = 0.0
                        acc = 0.0

                if it == 0 or it == (self.steps - 1):
                    self._log(f"[DIA] attack iter {it + 1}/{self.steps}")
                    self._log(f"[DIA.train] step={it + 1}/{self.steps} loss={loss_val:.6f} benign_acc={acc:.4f} avg_conf={avg_conf:.6f}")

            final_adv = self.tta_attack_update(images)

        finally:
            # must restore persistent state
            try:
                self._restore_model_after_attack(verify_restore=debug_check_restore and self.debug)
            except Exception as e:
                self._log("[DIA] warning: restore failed:", e, level='warning')

            # optional debug compare of logits if requested
            if debug_check_restore and pre_logits is not None and self.debug:
                try:
                    was_training = self.model.training
                    self.model.eval()
                    for m in self.model.modules():
                        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                            m.track_running_stats = True
                    with torch.no_grad():
                        post_logits = self.model(images.to(self.device)).detach().cpu()
                    if was_training:
                        self.model.train()
                    else:
                        self.model.eval()
                    diff = (pre_logits - post_logits).abs().max().item()
                    if diff > 1e-4:
                        self._log(f"[DIA debug] restore mismatch: max diff={diff:.6e}", level='warning')
                    else:
                        self._log(f"[DIA debug] restore OK: diff={diff:.6e}")
                except Exception as e:
                    self._log("[DIA debug] failed check:", e, level='warning')

        # create is_adv mask
        is_adv_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        for idx in self.attack_index:
            is_adv_mask[idx] = True

        # stats & validate
        stats = self.validate_attack_effectiveness(images, targets, final_adv, is_adv_mask)
        self.attack_stats['total_attacks'] += num_attack
        self.attack_stats['successful_attacks'] += int(stats.get('attack_success_rate', 0.0) * num_attack)
        self.attack_stats['confidence_reduction'] += stats.get('confidence_reduction', 0.0) * stats.get('attack_samples', 0)
        self.attack_stats['accuracy_drops'].append(stats['benign_accuracy_drop'])
        # optional save few adv examples
        if self._make_dirs:
            adv_save_dir = os.path.join("results", "adv_samples")
            os.makedirs(adv_save_dir, exist_ok=True)
            saved = 0
            for idx in range(B):
                if is_adv_mask[idx] and saved < 3:
                    try:
                        def denorm(x):
                            return torch.clamp(x * self.norm_std + self.norm_mean, 0.0, 1.0)
                        orig = denorm(images[idx:idx+1])[0].cpu().numpy().transpose(1, 2, 0) * 255.0
                        adv = denorm(final_adv[idx:idx+1])[0].cpu().numpy().transpose(1, 2, 0) * 255.0
                        orig_img = Image.fromarray(np.uint8(orig.clip(0,255)))
                        adv_img = Image.fromarray(np.uint8(adv.clip(0,255)))
                        batch_num = (self.attack_stats['total_attacks'] // max(1, num_attack))
                        orig_path = os.path.join(adv_save_dir, f"batch{batch_num}_idx{idx}_orig.png")
                        adv_path = os.path.join(adv_save_dir, f"batch{batch_num}_idx{idx}_adv.png")
                        orig_img.save(orig_path)
                        adv_img.save(adv_path)
                        self._log(f"[DIA] saved examples: {orig_path}, {adv_path}")
                        saved += 1
                    except Exception as e:
                        self._log("[DIA] warning: failed to save adv example:", e, level='warning')

        if len(iter_losses) > 0:
            self._log(f"[DIA] Iteration losses summary count={len(iter_losses)} mean={float(np.mean(iter_losses)):.6f} min={float(np.min(iter_losses)):.6f} max={float(np.max(iter_losses)):.6f}")

        self._log(f"[DIA] Finished attack: attack_success_rate={stats.get('attack_success_rate',0.0)*100:.2f}%, benign_drop={stats.get('benign_accuracy_drop',0.0):.6f}")

        return final_adv, is_adv_mask

    def get_attack_stats(self):
        total = self.attack_stats['total_attacks']
        succ = self.attack_stats['successful_attacks']
        if total > 0:
            success_rate = succ / total
            avg_conf_red = self.attack_stats['confidence_reduction'] / max(1, total)
            avg_acc_drop = np.mean(self.attack_stats['accuracy_drops']) if len(self.attack_stats['accuracy_drops']) > 0 else 0.0
        else:
            success_rate = 0.0
            avg_conf_red = 0.0
            avg_acc_drop = 0.0
        return {
            'total_attacks': total,
            'successful_attacks': succ,
            'success_rate': success_rate,
            'avg_confidence_reduction': avg_conf_red,
            'avg_accuracy_drop': avg_acc_drop
        }




# ==============================================================================
# [Step 2] Data Loading Module (Replaces Original MixedDataset)
# ==============================================================================

class CIFAR100CDataset(torch.utils.data.Dataset):
    """
    Dataset loader for CIFAR-100-C in .npy format.
    Also supports directory-structured adversarial examples.
    Filenames are sorted for deterministic loading.
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

            if level is not None and corruption_type is not None and corruption_type.lower() not in ['clean', 'none']:
                start_idx = (level - 1) * 10000
                end_idx = level * 10000
                if start_idx < len(self.data):
                    self.data = self.data[start_idx:min(end_idx, len(self.data))]
                    self.targets = self.targets[start_idx:min(end_idx, len(self.targets))]
                else:
                    raise ValueError(f"Corruption level {level} out of range.")

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


class MixedDataset(torch.utils.data.Dataset):
    """
    Dataset that combines clean and adversarial samples.
    Returns: (image, label, is_adversarial)
    """
    def __init__(self, clean_dataset, adv_dataset, adv_ratio=0.0, use_adv=True):
        self.clean_dataset = clean_dataset
        self.adv_dataset = adv_dataset
        self.adv_ratio = adv_ratio if use_adv else 0.0
        self.use_adv = use_adv

        assert len(clean_dataset.classes) == len(adv_dataset.classes), "Dataset class counts do not match"

        self.clean_indices_by_class = defaultdict(list)
        for idx in range(len(clean_dataset)):
            label = clean_dataset.targets[idx]
            self.clean_indices_by_class[label].append(idx)

        self.adv_indices_by_class = defaultdict(list)
        for idx in range(len(adv_dataset)):
            label = adv_dataset.targets[idx]
            self.adv_indices_by_class[label].append(idx)

        self.length = len(clean_dataset)
        self.classes = clean_dataset.classes
        
        self.benign_indices = list(range(self.length))
        self.adv_indices = []
        self.adv_mapping = {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx in self.adv_mapping:
            adv_idx = self.adv_mapping[idx]
            img, label = self.adv_dataset[adv_idx]
            return img, label, True  
        else:
            img, label = self.clean_dataset[idx]
            return img, label, False 


class FixedAdversarialBatchSampler(torch.utils.data.Sampler):
    """
    Deterministic batch sampler for mixed clean-adversarial datasets.
    Uses local random generator for full reproducibility.
    """
    def __init__(self, dataset, batch_size, adv_ratio=0.2, drop_last=False, seed=2025): 
        self.dataset = dataset
        self.batch_size = batch_size
        self.adv_ratio = adv_ratio
        self.drop_last = drop_last
        self.seed = seed 

        self.adv_per_batch = int(batch_size * adv_ratio)
        self.clean_per_batch = batch_size - self.adv_per_batch

        self._prepare_adv_samples()

    def _prepare_adv_samples(self):
        dataset = self.dataset
        dataset.benign_indices = list(range(len(dataset.clean_dataset)))
        dataset.adv_indices = []
        dataset.adv_mapping = {}

        required_adv = int(len(dataset) * self.adv_ratio)

        rng = random.Random(self.seed) 
        
        if dataset.use_adv and self.adv_ratio > 0:
            adv_indices = rng.sample(dataset.benign_indices, required_adv)

            for idx in adv_indices:
                label = dataset.clean_dataset.targets[idx]
                if dataset.adv_indices_by_class[label]:
                    adv_idx = rng.choice(dataset.adv_indices_by_class[label])
                    dataset.adv_mapping[idx] = adv_idx
                    dataset.adv_indices.append(idx)
                    dataset.benign_indices.remove(idx)

    def __iter__(self):

        rng = random.Random(self.seed)

        benign_indices = self.dataset.benign_indices.copy()
        adv_indices = self.dataset.adv_indices.copy()
        
        
        rng.shuffle(benign_indices)
        rng.shuffle(adv_indices)

        if self.adv_per_batch > 0 and self.clean_per_batch > 0:
            total_batches = min(
                len(benign_indices) // self.clean_per_batch,
                len(adv_indices) // self.adv_per_batch
            )
        else:
            total_batches = len(benign_indices) // self.batch_size

        for i in range(total_batches):
            batch_indices = []
            
            if self.adv_per_batch > 0:
                start = i * self.adv_per_batch
                end = (i + 1) * self.adv_per_batch
                batch_indices.extend(adv_indices[start:end])

            if self.clean_per_batch > 0:
                start = i * self.clean_per_batch
                end = (i + 1) * self.clean_per_batch
                batch_indices.extend(benign_indices[start:end])

            
            rng.shuffle(batch_indices)
            yield batch_indices

        if not self.drop_last:
            remaining_adv = []
            remaining_benign = []
            if self.adv_per_batch > 0:
                remaining_adv = adv_indices[total_batches * self.adv_per_batch:]
            if self.clean_per_batch > 0:
                remaining_benign = benign_indices[total_batches * self.clean_per_batch:]
            
            if remaining_adv or remaining_benign:
                last_batch = remaining_adv + remaining_benign
                rng.shuffle(last_batch)
                yield last_batch

    def __len__(self):
        if self.drop_last:
             if self.adv_per_batch > 0:
                return min(
                    len(self.dataset.benign_indices) // self.clean_per_batch,
                    len(self.dataset.adv_indices) // self.adv_per_batch
                )
             return len(self.dataset.benign_indices) // self.batch_size
        else:
             return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def prepare_model(config, logger, class_num):
    """
    Initialize and load pre-trained model for test-time adaptation.
    """
    logger.info(f"Model: {config['model']}")
    
    if config['model'] == "resnet26":
        logger.info("Initializing ResNet26 (CIFAR variant)...")

        import torch.nn as nn
        
        class BasicBlock(nn.Module):
            expansion = 1
            def __init__(self, in_planes, planes, stride=1):
                super(BasicBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)

                # Shortcut connection for dimension matching
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
            """ResNet26 model optimized for CIFAR image classification"""
            def __init__(self, num_classes=10):
                super(ResNet26_CIFAR, self).__init__()
                self.in_planes = 64
                block = BasicBlock
                num_blocks = [3, 3, 3, 3]  
                
                # CIFAR-specific input stem
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                
                self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
                self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
                self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
                self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
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

        # Initialize model with target number of classes
        model_backbone = ResNet26_CIFAR(num_classes=class_num)
        
        # Load pre-trained weights
        pretrained_path = "resnet26_cifar10_finetuned.pth" 
        
        if not os.path.exists(pretrained_path): 
            pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resnet26_cifar100_finetuned.pth")

        if os.path.exists(pretrained_path):
            logger.info(f"加载 ResNet26 权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle checkpoint formats with state_dict or DataParallel wrapping
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") 
                new_state_dict[name] = v
            
            model_backbone.load_state_dict(new_state_dict, strict=True)
        else:
            logger.warning(f"Pre-trained weights not found at {pretrained_path}, using random initialization")

        # Move model to specified device
        model_backbone = model_backbone.to(torch.device(f"cuda:{config['gpu']}" if config['gpu'] is not None else "cpu"))
        lr = 0.001       

    else:
        logger.error(f"Unsupported model type: {config['model']}")
        raise NotImplementedError

    # Adjust final fully connected layer while preserving pre-trained weights
    if hasattr(model_backbone, 'fc'):
        original_fc = model_backbone.fc
        if class_num <= 1000:  
            import torch.nn as nn
            new_fc = nn.Linear(original_fc.in_features, class_num)
            with torch.no_grad():
                new_fc.weight.copy_(original_fc.weight[:class_num])
                new_fc.bias.copy_(original_fc.bias[:class_num])
            model_backbone.fc = new_fc
            logger.info(f"Adjusted output layer to {class_num} classes with preserved pre-trained weights")


    # Apply Median Batch Normalization if enabled
    if config['medbn']['enable']:
        from rbn import RBN
        logger.info("Applying Median Batch Normalization for TTA")
        model_backbone = RBN.adapt_model(
            model_backbone,
            prior=config['medbn']['prior']
        )
        logger.info(f"MedBN applied with prior={config['medbn']['prior']}")

    # Configure test-time adaptation method
    logger.info(f"Configuring TTA method: {config['method']}")
    if config['method'] == 'no_adapt':
        logger.info("Using baseline model without gradient adaptation")
        model = model_backbone
    elif config['method'] == 'tent':
        logger.info("Initializing Tent for test-time adaptation")

        model = tent.configure_model(model_backbone)
        params, param_names = tent.collect_params(model)
        logger.info(f"Tent optimizing {len(params)} parameters:")
        for name in param_names:
            logger.info(f"  - {name}")
        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
        model = tent.Tent(model, optimizer)

    elif config['method'] == 'eata':
        logger.info("Initializing EATA for test-time adaptation")
        model = eata.configure_model(model_backbone)
        params, param_names = eata.collect_params(model)

        logger.info(f"EATA optimizing {len(params)} parameters:")
        for name in param_names:
            logger.info(f"  - {name}")
        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
        model = eata.EATA(model, optimizer, e_margin=config['e_margin'],
                          d_margin=config['d_margin'], fisher_alpha=config['fisher_alpha'])
    
    elif config['method'] == 'sar':
        logger.info("Initializing SAR for test-time adaptation")
        model = sar.configure_model(model_backbone)
        params, param_names = sar.collect_params(model)
        base_optimizer = torch.optim.SGD
        logger.info(f"SAR optimizing {len(params)} parameters:")
        for name in param_names:
            logger.info(f"  - {name}")
        optimizer = SAM(params, base_optimizer, lr=lr, momentum=0.9)
        model = sar.SAR(model, optimizer, steps=1, episodic=False, margin_e0=config['sar_margin_e0'])

    elif config['method'] == 'sotta':
        logger.info("Initializing SoTTA for test-time adaptation")

        import medbn as _local_medbn  
        sotta_module.MedBN = getattr(_local_medbn, "MedBN", None)
        sotta_module._MEDBN = True if sotta_module.MedBN is not None else False

        logger.info(f"MedBN integration status: _MEDBN={sotta_module._MEDBN}, MedBN={sotta_module.MedBN}")

        # Configure model and collect optimization parameters
        model_for_sotta = sotta_module.configure_model(model_backbone)
        params, param_names = sotta_module.collect_params(model_for_sotta)

        logger.info(f"SoTTA optimizing {len(params)} parameters:")
        for name in param_names:
            logger.info(f"  - {name}")

        # Initialize optimizer with SAM fallback
        base_optimizer = torch.optim.SGD
        try:
            # try using your SAM implementation (sam imported earlier)
            optimizer = SAM(params, base_optimizer, lr=lr, momentum=0.9)
            logger.info("Using SAM optimizer for SoTTA")
        except Exception as e:
            logger.warning(f"SAM initialization failed: {str(e)}, falling back to SGD")
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

        # Create SoTTA instance from configuration
        try:
            sota_obj = sotta_module.SoTTA.from_config(
                model=model_for_sotta,
                optimizer=optimizer,
                config=config,          
                feature_extractor=None
            )
            model = sota_obj
            logger.info("Successfully initialized SoTTA")
        except Exception as e:
            logger.error(f"Failed to initialize SoTTA: {str(e)}")
            logger.error("Falling back to baseline model")
            model = model_for_sotta
    # Create frozen copy of original model for feature extraction
    import copy
    raw_model_for_copy = model.model if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module) else model
    model_orig_for_features = copy.deepcopy(raw_model_for_copy).cuda(config['gpu']) if config['gpu'] is not None else copy.deepcopy(raw_model_for_copy)
    model_orig_for_features.eval()

    return model, model_orig_for_features


def load_mixed_data(config, logger):
    """
    Load mixed CIFAR-100-C dataset with optional adversarial samples.
    Creates deterministic dataloaders with reproducible randomness.
    
    Args:
        config: Configuration dictionary containing dataset paths and parameters
        logger: Logger instance for dataset loading status
        
    Returns:
        test_loader: Configured DataLoader for mixed test dataset
    """
    cifar_root = config['cifar100c_root']
    logger.info(f"Loading CIFAR-100-C from root directory: {cifar_root}")
    
    # Worker initialization function for deterministic multi-processing
    def seed_worker(worker_id):
        worker_seed = (torch.initial_seed() + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Random generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(config['seed'])

    # Load clean/corrupted CIFAR-100-C dataset
    try:
        clean_dataset = CIFAR100CDataset(
            root=cifar_root,
            corruption_type=config['corruption_type'],
            level=config['corruption_level'],
            transform=test_transforms,
            is_adv=False
        )
    except Exception as e:
        logger.error(f"Failed to load primary dataset: {e}")
        raise

    if config['use_adv']:
        adv_dataset = CIFAR100CDataset(root=config['adv_data'], transform=test_transforms, is_adv=True)
        mixed_dataset = MixedDataset(clean_dataset, adv_dataset, adv_ratio=config['adv_ratio'], use_adv=True)
    else:
        mixed_dataset = MixedDataset(clean_dataset, clean_dataset, adv_ratio=0.0, use_adv=False)

    # Create DataLoader with deterministic configuration
    if config['use_adv'] and config['adv_ratio'] > 0:
        batch_sampler = FixedAdversarialBatchSampler(
            mixed_dataset,
            batch_size=config['test_batch_size'],
            adv_ratio=config['adv_ratio'],
            drop_last=False,
            seed=config['seed']  
        )
        test_loader = torch.utils.data.DataLoader(
            mixed_dataset,
            batch_sampler=batch_sampler,
            num_workers=config['workers'],
            pin_memory=True,
            worker_init_fn=seed_worker, 
            generator=g                 
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            mixed_dataset,
            batch_size=config['test_batch_size'],
            shuffle=True,
            num_workers=config['workers'],
            pin_memory=True,
            worker_init_fn=seed_worker, 
            generator=g                 
        )
        
    return test_loader


def evaluate(test_loader, model, model_orig_for_features, config, logger):
    """
    Evaluate model performance on corrupted/adversarial data with test-time adaptation.
    Supports DIA attack generation, adversarial detection, and various TTA strategies.
    
    Args:
        test_loader: DataLoader for test dataset containing clean/adversarial samples
        model: Adaptable model for test-time optimization
        model_orig_for_features: Frozen pre-trained model for feature extraction
        config: Configuration dictionary for evaluation, attacks, and TTA
        logger: Logger instance for tracking evaluation progress and metrics
    
    Returns:
        Tuple containing average top-1 accuracy, error rate, AUROC, FPR@TPR95,
        adversarial filter instance, and DIA attacker instance
    """
    batch_time = AverageMeter('time', ':6.3f')

    # Evaluation metrics for benign samples only
    clean_top1 = AverageMeter('Benign Top-1 Acc', ':6.2f')
    er_metric = AverageMeter('Benign Error Rate', ':6.2f')
    auroc_meter = AverageMeter('Benign AUROC', ':6.2f')
    fpr_at_tpr95_meter = AverageMeter('Benign FPR@TPR95', ':6.2f')

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, clean_top1, er_metric, auroc_meter, fpr_at_tpr95_meter],
        prefix='Test: '
    )

    # Initialize DIA attacker if enabled
    dia_config = config['dia_attack']
    dia_attacker = None
    if dia_config['enabled']:
        raw_model_for_attack = model.model if hasattr(model, 'model') else model
        dia_attacker = DIAAttacker(model=raw_model_for_attack, config=config, logger=logger)
        logger.info(f"Initialized DIA attacker: type={dia_config['attack_type']}, ratio={dia_config['attack_ratio']}")

    # Initialize adversarial detection filter
    adv_filter_config = config.get('adv_detection', {})
    adv_filterer = adv_filter.AdvFilter(
        window_size=adv_filter_config.get('window_size', 10),
        threshold_method=adv_filter_config.get('threshold_method', 'std'),
        quantile_val=adv_filter_config.get('quantile', 0.8),
        std_factor=adv_filter_config.get('std_factor', 1.0),
        weights=adv_filter_config.get('feature_weights', [0.25] * 4),
        logger=logger,
        device=torch.device(f"cuda:{config['gpu']}") if config['gpu'] is not None and torch.cuda.is_available() else torch.device('cpu'),

    )


    end = time.time()
    for batch_idx, (images, targets, original_is_adv) in enumerate(test_loader):
        try:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Batch {batch_idx + 1}/{len(test_loader)}")
            logger.info(f"{'=' * 80}")
            # Move data to specified device
            if config['gpu'] is not None:
                images = images.cuda(config['gpu'], non_blocking=True)
                targets = targets.cuda(config['gpu'], non_blocking=True)
                if original_is_adv is not None:
                    original_is_adv = original_is_adv.cuda(config['gpu'], non_blocking=True)

            # Step 1: Generate DIA adversarial examples
            logger.info("\n[1. DIA Adversarial Attack]")
            if dia_config['enabled']:
                attacked_images, dia_is_adv = dia_attacker.generate_dia_attack(
                    images, targets,
                    attack_ratio=dia_config['attack_ratio'],
                    logger=logger
                )

                images = attacked_images
                is_adv = dia_is_adv

                num_adv = dia_is_adv.sum().item()
                logger.info(f"Generated {num_adv} DIA adversarial samples ({num_adv / len(images):.2%})")
                logger.info(f"Attack params: type={dia_config['attack_type']}, eps={dia_config['eps']}, steps={dia_config['steps']}")

            else:
                is_adv = original_is_adv if original_is_adv is not None else torch.zeros_like(targets, dtype=torch.bool)
                logger.info("DIA attack disabled, using original dataset adversarial labels")

            # Identify benign samples (non-adversarial)
            benign_mask = ~is_adv
            num_benign = benign_mask.sum().item()
            logger.info(f"Benign samples: {num_benign}, Adversarial samples: {len(images) - num_benign}")
            
            # Step 2: Adversarial detection and initial prediction
            images_after_detection_and_purification = images.clone()
            raw_model_for_prediction = model.model if hasattr(model, 'model') else model
            raw_model_for_prediction.eval()
            # Ensure batch normalization uses running statistics

            for module in raw_model_for_prediction.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.track_running_stats = True

            detected_by_advfilter_mask = torch.zeros_like(targets, dtype=torch.bool)
            tp_indices = torch.tensor([], dtype=torch.long, device=targets.device)
            tp_acc_before, tp_acc_after = -1.0, -1.0

            if adv_filter_config.get('enabled', False):
                logger.info("\n[2. Adversarial Detection]")
                _, _, _, detected_by_advfilter_mask, all_outputs = adv_filterer.filter_batch(
                    images=images,
                    targets=targets,
                    is_adv=is_adv,
                    config=config,
                    model_tta=raw_model_for_prediction,
                    model_orig=model_orig_for_features,
                    logger=logger,
                    batch_idx=batch_idx
                )
                num_detected = detected_by_advfilter_mask.sum().item()
                logger.info(f"Detected {num_detected} potential adversarial samples")
            else:
                logger.info("\n[A.1. Adversarial detection disabled]")
                # Manually forward pass to obtain logits for evaluation
                with torch.no_grad():
                    all_outputs = raw_model_for_prediction(images)

             # Calculate true positive detection accuracy
            if adv_filter_config.get('enabled', False):
                tp_mask = detected_by_advfilter_mask & is_adv
                tp_indices = tp_mask.nonzero(as_tuple=True)[0]
                if tp_indices.size(0) > 0:
                    with torch.no_grad():
                        tp_images_before = images[tp_indices]
                        tp_targets = targets[tp_indices]
                        tp_output_before = model_orig_for_features(tp_images_before)
                        _, tp_predicted_before = torch.max(tp_output_before, 1)
                        tp_correct_before = (tp_predicted_before == tp_targets)
                        tp_correct_before_count = tp_correct_before.sum().item()
                        tp_acc_before = 100.0 * tp_correct_before_count / tp_indices.size(0)


            # Step 3: Evaluate performance on benign samples
            logger.info("\n[3. Performance Evaluation (using model before TTA update)]")

            benign_count = benign_mask.sum().item()
            logger.info(f"Benign samples: {benign_count}, Adversarial samples: {is_adv.sum().item()}")

            if benign_count > 0:
                
                benign_output = all_outputs[benign_mask]
                benign_targets = targets[benign_mask]

                acc1, _ = accuracy(benign_output, benign_targets, topk=(1, 5))
                benign_acc = acc1[0].item()
                benign_er = 100.0 - benign_acc

                clean_top1.update(benign_acc, benign_count)
                er_metric.update(benign_er, benign_count)

                logger.info(f"Benign accuracy: {benign_acc:.2f}% (Error rate: {benign_er:.2f}%)")

                # Compute AUROC and FPR
                _, predicted = torch.max(benign_output, 1)
                is_correct = (predicted == benign_targets)
                probs = torch.softmax(benign_output, dim=1)
                confidences, _ = torch.max(probs, dim=1)

                y_true_roc = is_correct.cpu().numpy()
                y_score_roc = confidences.cpu().numpy()

                if len(np.unique(y_true_roc)) > 1:
                    try:
                        current_auroc = roc_auc_score(y_true_roc, y_score_roc) * 100
                        auroc_meter.update(current_auroc, benign_count)

                        fpr, tpr, _ = roc_curve(y_true_roc, y_score_roc)
                        if np.max(tpr) >= 0.95:
                            idx = np.argmin(np.abs(tpr - 0.95))
                            current_fpr_at_tpr95 = fpr[idx] * 100
                        else:
                            current_fpr_at_tpr95 = 100.0

                        fpr_at_tpr95_meter.update(current_fpr_at_tpr95, benign_count)

                        logger.info(f"Benign AUROC: {current_auroc:.2f}")
                        logger.info(f"Benign FPR@TPR95: {current_fpr_at_tpr95:.2f}")
                    except Exception as e:
                        logger.warning(f"Error computing AUROC/FPR@TPR95: {str(e)}")
                else:
                    logger.info("All benign samples classified correctly/incorrectly, skipping AUROC/FPR calculation")
            else:
                logger.warning("No benign samples in current batch, skipping evaluation")

            # Step 4: Sample filtering before TTA
            images_for_tta = images_after_detection_and_purification
            if adv_filter_config.get('enabled', False) and detected_by_advfilter_mask.any():
                if config['sample_filtering']['enabled']:
                    logger.info("\n[C.1. Sample Filtering]")
                    clean_indices = (~detected_by_advfilter_mask).nonzero(as_tuple=True)[0]
                    if clean_indices.size(0) > 0:
                        images_for_tta = images_after_detection_and_purification[clean_indices]
                        logger.info(f"Filtered out {detected_by_advfilter_mask.sum().item()} suspicious samples, {clean_indices.size(0)} samples remain for TTA.")
                    else:
                        images_for_tta = images_after_detection_and_purification  
                        logger.info("Warning: No samples left after filtering; using original batch for TTA.")

            # Step 5: Test-Time Adaptation (prepare for next batch)
            if config['method'] != 'no_adapt':
                logger.info("\n[D. Test-Time Adaptation (prepare for next batch)]")
                if images_for_tta.size(0) > 0:
                    if torch.isnan(images_for_tta).any() or torch.isinf(images_for_tta).any():
                        logger.warning("Invalid values detected in TTA inputs; applying preprocessing.")
                        images_for_tta = torch.nan_to_num(
                            images_for_tta,
                            nan=0.0,
                            posinf=1.0,
                            neginf=-1.0
                        )

                    # Collect normalization layers for state monitoring
                    monitored_layers = []
                    from types import SimpleNamespace
                    MAX_MONITORED = 6  
                    if hasattr(model, 'model'):
                    
                        for name, module in model.model.named_modules():
                            if config['model'].startswith('vit') and isinstance(module, torch.nn.LayerNorm):
                                monitored_layers.append((name, module, 'LayerNorm'))
                            elif not config['model'].startswith('vit') and isinstance(module, torch.nn.BatchNorm2d):
                                monitored_layers.append((name, module, 'BatchNorm2d'))
                            else:
        
                                if module.__class__.__name__ == 'RBN':
                                    monitored_layers.append((name, module, 'RBN'))
                            if len(monitored_layers) >= MAX_MONITORED:
                                break

                    # Record pre-TTA normalization layer states
                    pre_tta_states = {}
                    for name, module, mtype in monitored_layers:
                        try:
                            if mtype == 'LayerNorm':
                                if getattr(module, 'weight', None) is not None and getattr(module, 'bias', None) is not None:
                                    pre_tta_states[name] = {
                                        'type': mtype,
                                        'weight_mean': module.weight.detach().clone(),
                                        'bias_mean': module.bias.detach().clone(),
                                    }
                            elif mtype == 'BatchNorm2d':
                                pre_tta_states[name] = {
                                    'type': mtype,
                                    'running_mean': module.running_mean.detach().clone(),
                                    'running_var': module.running_var.detach().clone(),
                                    'weight': module.weight.detach().clone() if getattr(module, 'weight', None) is not None else None,
                                    'bias': module.bias.detach().clone() if getattr(module, 'bias', None) is not None else None
                                }
                            elif mtype == 'RBN':
                                # 优先取内部 layer
                                target = module.layer if hasattr(module, 'layer') else module
                                
                                pre_tta_states[name] = {
                                    'type': mtype,
                                    'running_mean': target.running_mean.detach().clone() if hasattr(target, 'running_mean') else None,
                                    'running_var': target.running_var.detach().clone() if hasattr(target, 'running_var') else None,
                                    'weight': target.weight.detach().clone() if hasattr(target, 'weight') and target.weight is not None else None,
                                    'bias': target.bias.detach().clone() if hasattr(target, 'bias') and target.bias is not None else None
                                }
                                
                        except Exception as e:
                            logger.warning(f"Error recording pre-TTA state for {name}: {e}")

                    logger.info(f"Adversarial ratio before TTA: {is_adv.sum().item() / len(is_adv):.2%}")
                    logger.info(f"Samples used for TTA: {images_for_tta.size(0)}")
                    logger.info(f"TTA iterations: {config['tta_iterations']}")
                    logger.info(f"Monitored norm layers: {', '.join([f'{n}:{t}' for n,_,t in monitored_layers])}")

                    # Set model to training mode for adaptation (disable dropout)
                    if hasattr(model, 'model'):
                        original_mode = model.model.training
                        model.model.train()
                        for m in model.model.modules():
                            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                                m.eval()

                    with torch.enable_grad():
                        for i in range(config['tta_iterations']):
                            logger.info(f"Running TTA iteration {i + 1}/{config['tta_iterations']}")
                            if hasattr(model, 'adapt') and callable(model.adapt):
                                model.adapt(images_for_tta)
                            else:
                                _ = model(images_for_tta)

                    if hasattr(model, 'model'):
                        model.model.train(original_mode)

                    # Compare layer states before and after TTA
                    if hasattr(model, 'model') and 'pre_tta_states' in locals() and len(monitored_layers) > 0:
                        logger.info("Norm layer parameter changes before/after TTA:")
                        for name, module, mtype in monitored_layers:
                            if name not in pre_tta_states:
                                continue
                            prev = pre_tta_states[name]
                            
                            try:
            
                                if prev['type'] == 'LayerNorm':
                                    post_weight = module.weight.detach().clone()
                                    post_bias = module.bias.detach().clone()
                                    weight_change = torch.abs(post_weight.mean() - prev['weight_mean'].mean()).item()
                                    bias_change = torch.abs(post_bias.mean() - prev['bias_mean'].mean()).item()
                                    logger.info(f"  - {name} (LayerNorm): weight_change={weight_change:.10f}, bias_change={bias_change:.10f}")
                                    
                                    if weight_change < 1e-9 and bias_change < 1e-9:
                                        logger.warning(f"    [Warning] {name} parameters unchanged - check LR/gradients for Tent/EATA.")

                               
                                elif prev['type'] == 'BatchNorm2d':
                                    post_mean = module.running_mean.detach().clone()
                                    post_var = module.running_var.detach().clone()
                                    mean_change = torch.abs(post_mean.mean() - prev['running_mean'].mean()).item()
                                    var_change = torch.abs(post_var.mean() - prev['running_var'].mean()).item()
                                    
    
                                    if module.affine and module.weight is not None:
                                        weight_change = torch.abs(module.weight.detach().mean() - prev['weight'].mean()).item()
                                        bias_change = torch.abs(module.bias.detach().mean() - prev['bias'].mean()).item()
                                    else:
                                        weight_change, bias_change = 0.0, 0.0

                                    logger.info(f"  - {name} (BN2d): stats_change(u/v)={mean_change:.6f}/{var_change:.6f}, affine_change(w/b)={weight_change:.8f}/{bias_change:.8f}")
                                    
                                    if mean_change < 1e-6 and var_change < 1e-6 and weight_change < 1e-9:
                                        logger.warning(f"    [Warning] {name} stats/parameters unchanged.")

                                
                                elif prev['type'] == 'RBN':
        
                                    real_layer = module.layer if hasattr(module, 'layer') else module
                                    
                                
                                    if hasattr(real_layer, 'weight') and real_layer.weight is not None and prev['weight'] is not None:
                                        post_w = real_layer.weight.detach()
                                        post_b = real_layer.bias.detach() if real_layer.bias is not None else torch.zeros_like(post_w)
                                        
                                        
                                        weight_change = torch.abs(post_w.mean() - prev['weight'].mean()).item()
                                        bias_change = torch.abs(post_b.mean() - prev['bias'].mean()).item() if prev['bias'] is not None else 0.0
                                    else:
                                        weight_change, bias_change = 0.0, 0.0

                            
                                    mean_change, var_change = 0.0, 0.0
                                    
                                    
                                    if hasattr(real_layer, 'source_mean') and hasattr(real_layer, 'source_var'):
                                        
                                        pass
                                    
                                   
                                    if hasattr(real_layer, 'running_mean') and real_layer.running_mean is not None and prev['running_mean'] is not None:
                                        mean_change = torch.abs(real_layer.running_mean.detach().mean() - prev['running_mean'].mean()).item()
                                        var_change = torch.abs(real_layer.running_var.detach().mean() - prev['running_var'].mean()).item()

                                    logger.info(f"  - {name} (MedBN): stats_change={mean_change:.6f}, affine_change(w/b)={weight_change:.8f}/{bias_change:.8f}")

                                    
                                    if config['method'] in ['tent', 'eata', 'sar']:
                                        
                                        if weight_change < 1e-9:
                                            logger.warning(f"    [Warning] {name} parameters not updated (LR={config.get('lr', 'Unknown')}), TTA may fail.")
                                    else:
                                        
                                        if mean_change < 1e-6:
                                            logger.info(f"    [Info] {name} stats buffer unchanged (normal for dynamic MedBN).")

                            except Exception as e:
                                logger.warning(f"Error comparing states for {name}: {e}")


                    logger.info(f"TTA completed on {images_for_tta.size(0)} samples for batch {batch_idx + 1}")

            # Step 6: Batch statistics
            batch_time.update(time.time() - end)
            end = time.time()

            logger.info("\n[E. Batch Statistics]")
            logger.info(f"Original batch size: {images.size(0)}")
            logger.info(f"Batch size for TTA: {images_for_tta.size(0)}")
            logger.info(f"Detected adversarial samples (original batch): {detected_by_advfilter_mask.sum().item()}")
            logger.info(f"True positive detections: {tp_indices.size(0)}")
            logger.info(f"Benign sample performance (before TTA):")
            if 'benign_acc' in locals():
                logger.info(f"  - Accuracy: {benign_acc:.2f}% (ER: {benign_er:.2f}%)")
            if 'current_auroc' in locals() and 'current_fpr_at_tpr95' in locals():
                logger.info(f"  - AUROC: {current_auroc:.2f}")
                logger.info(f"  - FPR@TPR95: {current_fpr_at_tpr95:.2f}")


            logger.info(f"Batch time: {batch_time.val:.2f}s")
            logger.info(f"{'=' * 80}\n")

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Final results
    logger.info(f"\nFinal Classification Results - Method: {config['method']}, Model: {config['model']}")
    logger.info(f"Final benign sample average performance:")
    logger.info(f"  - Accuracy: {clean_top1.avg:.2f}%")
    logger.info(f"  - Error Rate (ER): {er_metric.avg:.2f}%")
    logger.info(f"  - AUROC: {auroc_meter.avg:.2f}")
    logger.info(f"  - FPR@TPR95: {fpr_at_tpr95_meter.avg:.2f}")

    
        
    # Adversarial detection summary
    if config['adv_detection']['enabled'] and (config['use_adv'] or config['dia_attack']['enabled']):
        if hasattr(adv_filterer, 'log_detection_summary'):
            adv_filterer.log_detection_summary()

        else:
            logger.warning("adv_filterer has no log_detection_summary(); skipping detection stats.")

    # Feature distribution visualization
    
    logger.info("\nAnalyzing adversarial/benign feature distributions...")

    try:
        adv_filterer.output_dir = config.get('output', 'results')
        adv_filterer.plot_feature_distributions(output_dir=config.get('output', 'results'))
    except AttributeError:
        logger.warning("plot_feature_distributions() not found; skipping visualization.")
    except Exception as e:
        logger.error(f"Error plotting feature distributions: {e}")

    return clean_top1.avg, er_metric.avg, auroc_meter.avg, fpr_at_tpr95_meter.avg, adv_filterer, dia_attacker





def validate_config(config, logger):
    """Validate configuration parameters for correctness and consistency."""
    # Validate DIA attack configuration
    if config['dia_attack']['enabled']:
        dia_config = config['dia_attack']
        if not 0 <= dia_config['attack_ratio'] <= 1:
            logger.error("DIA attack ratio must be between 0 and 1")
            raise ValueError("attack_ratio must be between 0 and 1")
        if dia_config['eps'] <= 0:
            logger.error("DIA attack perturbation magnitude must be greater than 0")
            raise ValueError("eps must be greater than 0")
        if dia_config['steps'] <= 0:
            logger.error("DIA attack steps must be greater than 0")
            raise ValueError("steps must be greater than 0")
        if dia_config['attack_type'] not in ['indiscriminate', 'targeted', 'stealthy']:
            logger.error("DIA attack type must be 'indiscriminate', 'targeted', or 'stealthy'")
            raise ValueError("attack_type must be one of: 'indiscriminate', 'targeted', 'stealthy'")
        if dia_config['norm'] not in ['Linf', 'L2']:
            logger.error("DIA attack norm type must be 'Linf' or 'L2'")
            raise ValueError("norm must be one of: 'Linf', 'L2'")

    # Validate adversarial detection configuration
    if config['adv_detection']['enabled']:
        adv_config = config['adv_detection']

        # Validate feature weights sum to 1
        weights = adv_config['feature_weights']
        weight_sum = sum(weights)
        if not (0.99 <= weight_sum <= 1.01):
            logger.warning(f"Feature weights should sum to 1.0, current sum: {weight_sum:.2f}")

        



def _auto_type_cast(value_str):
    """Automatically convert string to appropriate data type."""
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    if value_str.lower() == 'none':
        return None
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def _update_config_from_args(config, args, logger):
    """Update configuration dictionary from command line arguments."""
    if not args.set:
        return

    logger.info("Overriding default config from command line:")
    for kv in args.set:
        try:
            key_path, value_str = kv.split('=', 1)
            value = _auto_type_cast(value_str)

            keys = key_path.split('.')
            d = config
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]

            d[keys[-1]] = value
            logger.info(f"  - {key_path} = {value}")
        except ValueError:
            logger.warning(f"Failed to parse argument '{kv}', use 'key=value' format.")



def main():
    """Main entry point for test-time adaptation evaluation pipeline."""
    # Use global configuration
    config = CONFIG

    # Configure logger
    dia_tag = f"dia{int(config['dia_attack']['attack_ratio'] * 100)}" if config['dia_attack']['enabled'] else "clean"
    log_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + \
               f"-{config['method']}-{config['model']}-{dia_tag}.txt"
    logger = get_logger(name="mixed_test", output_directory=config['output'], log_name=log_name, debug=False)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run test-time adaptation evaluation with configurable parameters.")
    parser.add_argument('--set',
                        metavar='KEY=VALUE',
                        nargs='+',
                        help="Override config values using dot notation (e.g., --set purification.enable=True adv_ratio=0.5)")
    args = parser.parse_args()

    # Update config from command line
    _update_config_from_args(config, args, logger)

    # Validate configuration
    validate_config(config, logger)

    # Set random seed for reproducibility

    if config['seed'] is not None:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True) 
           

    # Create output directory
    if not os.path.exists(config['output']):
        os.makedirs(config['output'], exist_ok=True)

    # Log current configuration
    logger.info("Current configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    logger.info(f"    {sub_key}:")
                    for ssub_key, ssub_value in sub_value.items():
                        logger.info(f"      {ssub_key}: {ssub_value}")
                else:
                    logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")

    # Configure device
    if config['gpu'] is not None and torch.cuda.is_available():
        logger.info(f"Using GPU: {config['gpu']}")
        torch.cuda.set_device(config['gpu'])
        logger.info(f"GPU Name: {torch.cuda.get_device_name(config['gpu'])}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(config['gpu']).total_memory / 1024 ** 3:.1f} GB")
    else:
        logger.info("Using CPU")

    try:
        # Load test dataset
        test_loader = load_mixed_data(config, logger)

        # Get number of classes from dataset
        num_classes = len(test_loader.dataset.classes)
        logger.info(f"Number of classes: {num_classes}")

        # Initialize model
        model, model_orig_for_features = prepare_model(config, logger, num_classes)

        if torch.cuda.is_available() and config['gpu'] is not None:
            model = model.cuda(config['gpu'])

        # Log model parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")

        # Run evaluation
        logger.info(f"Starting evaluation with {config['method']} method...")
        clean_acc, er, auroc, fpr, adv_filterer, dia_attacker = evaluate(test_loader, model, model_orig_for_features,
                                                                         config, logger)

        # Log final results
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Final Results - Method: {config['method']}, Model: {config['model']}")
        logger.info(f"{'=' * 80}")
        logger.info(f"Benign sample performance:")
        logger.info(f"  - Accuracy: {clean_acc:.2f}%")
        logger.info(f"  - Error Rate (ER): {er:.2f}%")
        logger.info(f"  - AUROC: {auroc:.2f}")
        logger.info(f"  - FPR@TPR95: {fpr:.2f}")

        

        if config['adv_detection']['enabled'] and (config['use_adv'] or config['dia_attack']['enabled']):
            if hasattr(adv_filterer, 'log_detection_summary'):
                adv_filterer.log_detection_summary()
            else:
                logger.warning("adv_filterer does not have log_detection_summary(), skipping detection stats.")

        # Save results to file
        results_file = os.path.join(config['output'], 'final_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Method: {config['method']}\n")
            f.write(f"Model: {config['model']}\n")
            f.write(f"Clean Accuracy: {clean_acc:.2f}%\n")
            f.write(f"Error Rate: {er:.2f}%\n")
            f.write(f"AUROC: {auroc:.2f}\n")
            f.write(f"FPR@TPR95: {fpr:.2f}\n")
            
        logger.info(f"Results saved to: {results_file}")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()