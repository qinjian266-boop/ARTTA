"""
SoTTA: Robust Test-Time Adaptation with Selective Optimization - Research Adaptation

REFERENCE:
    - Original Code: https://github.com/taeckyung/SoTTA
    - Original Paper: "SoTTA: Robust Test-Time Adaptation with Selective Optimization" (CVPR 2024)
    - Attack Model: Distribution Invading Attack (DIA) from "Uncovering Adversarial Risks of Test-Time Adaptation"

ADAPTATION LOGIC:
    This version is adapted to align with our specific adversarial evaluation pipeline 
    (Experiment 2.1: ARTTA vs. DIA Attack). 

    Key Adjustments:
    1. Pipeline Integration: Modified parameter collection and memory management to 
       support Median Batch Normalization (MedBN) and simulation of Distribution Invading Attacks (DIA).
    2. Consistency: Unified entropy calculation and optimization interfaces to ensure 
       a fair comparison between ARTTA and other TTA methods (Tent, EATA, SAR, SoTTA) 
       under malicious batch injection.

    Note: As the experimental protocol (attack type, batch size, etc.) differs from 
    the original SoTTA study, the performance results here are specific to this 
    adversarial robustness benchmark and may not match the original paper's optimal values.
"""

from copy import deepcopy
from collections import defaultdict
import math
import random
from typing import List, Tuple, Callable, Optional

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import os, sys

# --- Ensure project root is on sys.path so top-level modules (medbn.py) can be imported ---
_pkg_dir = os.path.dirname(os.path.abspath(__file__))  
_project_root = os.path.dirname(_pkg_dir)         
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Robust import logic with fallback mechanisms for various repository structures
try:
    from . import memory as memory_module
    from .sam_optimizer import SAM, sam_collect_params
    try:
        from .loss_functions import softmax_entropy as repo_softmax_entropy
        _repo_softmax_entropy = repo_softmax_entropy
    except Exception:
        _repo_softmax_entropy = None
    try:
        from .medbn import MedBN as _MedBNClass
        MedBN = _MedBNClass
        _MEDBN = True
    except Exception:
        _MEDBN = False
        MedBN = None
except Exception as err_rel:
    try:
        import memory as memory_module
        from sam_optimizer import SAM, sam_collect_params
        try:
            from loss_functions import softmax_entropy as repo_softmax_entropy
            _repo_softmax_entropy = repo_softmax_entropy
        except Exception:
            _repo_softmax_entropy = None
        try:
            from medbn import MedBN as _MedBNClass
            MedBN = _MedBNClass
            _MEDBN = True
        except Exception:
            _MEDBN = False
            MedBN = None
    except Exception as err:
        raise ImportError(f"Failed to load required SoTTA components: {err}")

# Prefer repo-provided softmax entropy if available
if _repo_softmax_entropy is not None:
    _softmax_entropy = _repo_softmax_entropy
else:
    @torch.jit.script
    def _softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)


# ---------------------------------------------------------
# State Management for Evaluation Consistency
# ---------------------------------------------------------
def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    opt_state = deepcopy(optimizer.state_dict()) if optimizer is not None else None
    return model_state, opt_state


def load_model_and_optimizer(model, optimizer, model_state, opt_state):
    model.load_state_dict(model_state, strict=True)
    if optimizer is not None and opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
        except Exception:
            # If optimizer/torch versions differ or device mapping differs, try a more permissive load
            optimizer_state = deepcopy(opt_state)
            # map tensors to the optimizer's param device if possible
            try:
                optimizer.load_state_dict(optimizer_state)
            except Exception:
                pass


# ---------------------------
# collect_params / configure_model / check_model
# ---------------------------
# Replace existing collect_params with this improved version
def collect_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """
    Identifies affine parameters in normalization layers. 
    Supports standard PyTorch layers and Median Batch Normalization (MedBN).
    """
    params = []
    names = []

    for nm, m in model.named_modules():
        prefix = nm if nm != '' else 'model'
        try:
            # Handle standard affine-based normalization layers
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
                for np_name, p in m.named_parameters(recurse=False):
                    if np_name in ['weight', 'bias'] and p is not None:
                        params.append(p)
                        names.append(f"{prefix}.{np_name}")

            else:
                # Support for MedBN (Median Batch Normalization) layers
                # MedBN is critical for resisting DIA as it uses robust median statistics
                clsname = m.__class__.__name__
                if clsname == 'MedBN' or (MedBN is not None and isinstance(m, MedBN)):
                    if hasattr(m, 'weight') and getattr(m, 'weight') is not None:
                        params.append(m.weight)
                        names.append(f"{prefix}.weight")
                    if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                        params.append(m.bias)
                        names.append(f"{prefix}.bias")
                    # Search for parameters in nested modules within MedBN wrappers
                    if hasattr(m, 'layer') and m.layer is not None:
                        try:
                            for np_name, p in m.layer.named_parameters(recurse=False):
                                if np_name in ['weight', 'bias'] and p is not None:
                                    params.append(p)
                                    names.append(f"{prefix}.layer.{np_name}")
                        except Exception:
                            pass
        except Exception:
            
            continue

    # Deduplication to ensure optimization stability
    uniq_params = []
    uniq_names = []
    seen = set()
    for p, n in zip(params, names):
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        uniq_params.append(p)
        uniq_names.append(n)

    if len(uniq_params) == 0:
        print("collect_params WARNING: no affine normalization parameters found.")
        print("Model norm modules summary (module_name, class, has_weight, has_bias):")
        for nm, m in model.named_modules():
            clsname = m.__class__.__name__
            has_weight = hasattr(m, 'weight') and getattr(m, 'weight') is not None
            has_bias = hasattr(m, 'bias') and getattr(m, 'bias') is not None
            print(f"  - {nm or 'model'}: {clsname}, has_weight={has_weight}, has_bias={has_bias}")
        raise ValueError("collect_params found no affine normalization parameters. "
                         "Check MedBN import/path or model conversion. See printed module summary above.")

    return uniq_params, uniq_names


def configure_model(model: nn.Module) -> nn.Module:
    """
    Prepares the model for TTA:
    - Sets train() mode to estimate batch statistics (required for BN/MedBN).
    - Freezes feature representation while enabling normalization adaptation.
    """
    model.train()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad_(True)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad_(True)
            # prefer batch stats during TTA
            try:
                m.track_running_stats = True
            except Exception:
                pass

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)

        # Enable MedBN parameters for robust distribution alignment
        elif MedBN is not None and isinstance(m, MedBN):
            if hasattr(m, 'weight') and getattr(m, 'weight') is not None:
                m.weight.requires_grad_(True)
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                m.bias.requires_grad_(True)
            # 若 medbn 中仍有 inner layer（老实现），开启 inner 的 affine
            if hasattr(m, 'layer') and m.layer is not None:
                for np_name, p in m.layer.named_parameters(recurse=False):
                    if np_name in ['weight', 'bias'] and p is not None:
                        p.requires_grad_(True)

    return model


def check_model(model: nn.Module):
    """Ensures model state is compatible with TTA requirements."""
    is_training = model.training
    assert is_training, "SoTTA needs train mode: call model.train()"
    grads = [p.requires_grad for p in model.parameters()]
    assert any(grads), "SoTTA needs params to update: check which require grad"
    assert not all(grads), "SoTTA should not update all params: check which require grad"
    has_norm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm))
                   or (_MEDBN and isinstance(m, MedBN)) for m in model.modules())
    assert has_norm, "SoTTA needs normalization layer parameters"


# ---------------------------
# InternalMemory wrapper
# ---------------------------
class InternalMemory:
    """
    Wrapper over repo memory implementations. Optionally accepts a feature_extractor(fn) that
    takes (model, xs_tensor) and returns features on CPU to store. If not provided we store raw inputs.
    """
    def __init__(self, mem_type='HUS', capacity=2048, hus_threshold=None, num_class=1000,
                 feature_extractor: Optional[Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]] = None,
                 model_for_feat: Optional[torch.nn.Module] = None):
        self.mem_type = str(mem_type).upper()
        self.num_class = int(num_class)
        self.feature_extractor = feature_extractor
        self.model_for_feat = model_for_feat
        if self.mem_type == 'HUS':
            self.mem = memory_module.HUS(capacity=capacity, threshold=hus_threshold)
        elif self.mem_type == 'CONFFIFO':
            self.mem = memory_module.ConfFIFO(capacity=capacity, threshold=hus_threshold if hus_threshold is not None else 0.0)
        else:
            self.mem = memory_module.FIFO(capacity=capacity)

    def add_instance_from_batch(self, xs: torch.Tensor, logits: torch.Tensor):
        """
        xs: Tensor [B,C,H,W] on device (or CPU)
        logits: Tensor [B, num_classes] on same device as xs
        Store features on CPU as repo expects. If feature_extractor provided, use it to extract features.
        """
        try:
            probs = F.softmax(logits.detach().cpu(), dim=1)
            confs, preds = probs.max(1)
            # extract features if extractor provided
            if self.feature_extractor is not None and self.model_for_feat is not None:
                try:
                    # run extractor without grad on appropriate device
                    device = next(self.model_for_feat.parameters()).device
                    xs_dev = xs.to(device)
                    with torch.no_grad():
                        feats = self.feature_extractor(self.model_for_feat, xs_dev)
                    feats_cpu = feats.detach().cpu()
                except Exception:
                    # fallback to storing input
                    feats_cpu = xs.detach().cpu()
            else:
                feats_cpu = xs.detach().cpu()

            B = feats_cpu.size(0)
            for i in range(B):
                feat = feats_cpu[i]
                pred = int(preds[i].item())
                conf = float(confs[i].item())
                domain = 0
                if isinstance(self.mem, memory_module.HUS):
                    inst = (feat, pred, domain, conf)
                elif isinstance(self.mem, memory_module.ConfFIFO):
                    inst = (feat, pred, domain, conf)
                else:
                    inst = (feat, pred, domain)
                self.mem.add_instance(inst)
        except Exception:
            # very defensive fallback
            try:
                xs_cpu = xs.detach().cpu()
                probs = F.softmax(logits.detach().cpu(), dim=1)
                confs, preds = probs.max(1)
                B = xs_cpu.size(0)
                for i in range(B):
                    feat = xs_cpu[i]
                    pred = int(preds[i].item())
                    conf = float(confs[i].item())
                    domain = 0
                    if isinstance(self.mem, memory_module.HUS):
                        inst = (feat, pred, domain, conf)
                    elif isinstance(self.mem, memory_module.ConfFIFO):
                        inst = (feat, pred, domain, conf)
                    else:
                        inst = (feat, pred, domain)
                    self.mem.add_instance(inst)
            except Exception:
                return

    def hus_sample_tensor(self, batch_size: int):
        try:
            if isinstance(self.mem, memory_module.HUS):
                feats, cls_list, dls = self.mem.get_memory()
                if len(feats) == 0:
                    return None
                if len(feats) >= batch_size:
                    per_cls = max(1, math.ceil(batch_size / float(self.num_class)))
                    selected = []
                    class_to_indices = defaultdict(list)
                    for idx, c in enumerate(cls_list):
                        class_to_indices[int(c)].append(idx)
                    for c in range(self.num_class):
                        idxs = class_to_indices.get(c, [])
                        if not idxs:
                            continue
                        take = min(len(idxs), per_cls)
                        selected.extend(random.sample(idxs, take))
                        if len(selected) >= batch_size:
                            break
                    if len(selected) < batch_size:
                        selected = list(range(max(0, len(feats) - batch_size), len(feats)))
                    xs = torch.stack([feats[i] for i in selected[:batch_size]])
                    return xs
                else:
                    return torch.stack(feats) if len(feats) > 0 else None
            else:
                memdata = self.mem.get_memory()
                feats = memdata[0]
                if len(feats) == 0:
                    return None
                if len(feats) >= batch_size:
                    xs = torch.stack(feats[-batch_size:])
                else:
                    xs = torch.stack(feats)
                return xs
        except Exception:
            return None


# ---------------------------
# ESM update helper
# ---------------------------
def esm_update_step(model: nn.Module, optimizer: torch.optim.Optimizer, xs_cpu: torch.Tensor,
                    rho: float = 0.01):
    """
    xs_cpu: CPU tensor [B,...]; move to device inside
    Returns: loss value (float)
    This function will try to temporarily set optimizer.rho if present to the provided rho.
    """
    device = next(model.parameters()).device
    xs = xs_cpu.to(device)
    model.train()
    logits = model(xs)
    loss = _softmax_entropy(logits).mean()

    # If optimizer supports rho attribute, temporarily set it
    original_rho = None
    if hasattr(optimizer, 'rho'):
        try:
            original_rho = getattr(optimizer, 'rho')
            setattr(optimizer, 'rho', rho)
        except Exception:
            original_rho = None

    if hasattr(optimizer, 'first_step') and hasattr(optimizer, 'second_step'):
        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)
        logits2 = model(xs)
        loss2 = _softmax_entropy(logits2).mean()
        loss2.backward()
        optimizer.second_step(zero_grad=True)
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # restore rho if we changed it
    if original_rho is not None and hasattr(optimizer, 'rho'):
        try:
            setattr(optimizer, 'rho', original_rho)
        except Exception:
            pass

    return float(loss.item())


# ---------------------------
# SoTTA wrapper
# ---------------------------
class SoTTA(nn.Module):
    """
    SoTTA wrapper.
    Args:
        model: nn.Module (backbone+head). configure_model() should be called before constructing SoTTA.
        optimizer: optimizer acting on normalization affine params
        steps: number of minibatch adaptation steps per forward
        episodic: whether to reset after each forward
        mem_type/mem_capacity/hus_threshold/hus_batch_size/esm_rho/num_classes: memory and esm settings
        feature_extractor: optional callable (model, xs) -> features tensor on device; used when storing features
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 steps: int = 1, episodic: bool = False,
                 mem_type: str = 'HUS', mem_capacity: int = 2048, hus_threshold: float = None,
                 hus_batch_size: int = 64, esm_rho: float = 0.01, num_classes: int = 1000,
                 feature_extractor: Optional[Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SoTTA requires >=1 step(s) to forward and update"
        self.episodic = episodic

        # internal memory may want a feature extractor and reference model
        model_for_feat = None
        if feature_extractor is not None:
            model_for_feat = model
        self.mem = InternalMemory(mem_type, capacity=mem_capacity, hus_threshold=hus_threshold,
                                  num_class=num_classes, feature_extractor=feature_extractor,
                                  model_for_feat=model_for_feat)
        self.hus_batch_size = hus_batch_size
        self.esm_rho = esm_rho
        self.num_classes = num_classes

        # save copies for possible reset
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x: torch.Tensor):
        if self.episodic:
            self.reset()

        # compute outputs without grad for storage/prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

        # add to internal memory (features moved to CPU inside)
        try:
            self.mem.add_instance_from_batch(x, outputs)
        except Exception:
            # fallback: ensure both are CPU
            self.mem.add_instance_from_batch(x.cpu(), outputs.cpu())

        # adaptation steps (sample from memory and run esm update)
        if self.steps > 0:
            for _ in range(self.steps):
                sample = self.mem.hus_sample_tensor(self.hus_batch_size)
                if sample is None:
                    break
                _ = esm_update_step(self.model, self.optimizer, sample, rho=self.esm_rho)

        return outputs

    def reset(self):
        if self.model_state is None:
            raise RuntimeError("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)


    @classmethod
    def from_config(cls, model: nn.Module, optimizer: torch.optim.Optimizer, config: dict,
                    feature_extractor: Optional[Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]] = None):
        """
        Construct a SoTTA instance from a configuration dict (main-process friendly).
        The config may contain a nested 'sotta' dict or top-level keys. Example keys read:
        - sotta.mem_type / mem_type
        - sotta.mem_capacity / mem_capacity
        - sotta.hus_batch_size / hus_batch_size
        - sotta.hus_threshold / hus_threshold
        - sotta.esm_rho / esm_rho / rho
        - sotta.steps / steps / tta_iterations
        - sotta.episodic / episodic
        - sotta.num_classes / num_classes

        This method makes it convenient for the main script to pass the global CONFIG directly.
        If feature_extractor is None, we will attempt to pick a sensible default for ResNet and ViT.
        """
        # prefer nested sotta block if present
        s = {}
        if isinstance(config, dict):
            s = config.get('sotta', {}) if 'sotta' in config else config
        # helper to lookup keys with fallbacks
        def _get(keys, default=None):
            for k in keys:
                if k in s:
                    return s[k]
                if k in config:
                    return config[k]
            return default

        mem_type = _get(['mem_type'], 'HUS')
        mem_capacity = _get(['mem_capacity'], 2048)
        hus_batch_size = _get(['hus_batch_size'], _get(['test_batch_size'], 64))
        hus_threshold = _get(['hus_threshold'], None)
        esm_rho = _get(['esm_rho', 'rho'], 0.01)
        steps = _get(['steps', 'tta_iterations'], 1)
        episodic = _get(['episodic'], False)
        num_classes = _get(['num_classes'], _get(['class_num', 'classes', 'num_classes'], 1000))

        # ------------------ default feature extractors ------------------
        def resnet_feature_extractor(model_resnet, x):
            """Extract features from torchvision ResNet-like models: output of avgpool flattened."""
            # runs standard forward up to avgpool
            # try to avoid changing model state
            was_training = model_resnet.training
            model_resnet.eval()
            with torch.no_grad():
                out = x
                # common attribute names used by torchvision resnet
                if hasattr(model_resnet, 'conv1'):
                    out = model_resnet.conv1(out)
                if hasattr(model_resnet, 'bn1'):
                    out = model_resnet.bn1(out)
                if hasattr(model_resnet, 'relu'):
                    out = model_resnet.relu(out)
                if hasattr(model_resnet, 'maxpool'):
                    out = model_resnet.maxpool(out)
                for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                    if hasattr(model_resnet, layer_name):
                        layer = getattr(model_resnet, layer_name)
                        out = layer(out)
                # avgpool -> flatten
                if hasattr(model_resnet, 'avgpool'):
                    out = model_resnet.avgpool(out)
                    out = torch.flatten(out, 1)
                else:
                    out = torch.flatten(out, 1)
            model_resnet.train(was_training)
            return out

        def vit_feature_extractor(model_vit, x):
            """Extract features from torchvision ViT-like models. Try forward_features or reasonable fallbacks."""
            was_training = model_vit.training
            model_vit.eval()
            with torch.no_grad():
                # preferred hook for timm/torchvision: forward_features
                if hasattr(model_vit, 'forward_features'):
                    feats = model_vit.forward_features(x)
                else:
                    # torchvision ViT may expose a method named 'get_intermediate_layers' or similar
                    try:
                        feats = model_vit.forward_features(x) if hasattr(model_vit, 'forward_features') else None
                    except Exception:
                        feats = None
                    if feats is None:
                        # fallback: run full forward and try to extract penultimate activations
                        out = model_vit(x)
                        feats = out
                # ensure 2D tensor
                if feats is not None:
                    if feats.dim() > 2:
                        feats = torch.flatten(feats, 1)
            model_vit.train(was_training)
            return feats

        def default_selector(model_obj):
            """Pick a reasonable default feature_extractor based on common model attributes."""
            # ResNet heuristic
            if hasattr(model_obj, 'avgpool') and hasattr(model_obj, 'layer4'):
                return resnet_feature_extractor
            # ViT heuristic
            if hasattr(model_obj, 'patch_embed') or hasattr(model_obj, 'encoder') or hasattr(model_obj, 'heads') or hasattr(model_obj, 'forward_features'):
                return vit_feature_extractor
            # fallback: no extractor (store raw inputs)
            return None

        # if caller didn't provide an extractor, only auto-select when explicitly enabled in config
        # default: do NOT auto-extract features (store raw input images). To enable feature storing,
        # set config['sotta']['store_features'] = True or top-level config['store_features'] = True.
        store_features_flag = bool(s.get('store_features', False) or (isinstance(config, dict) and config.get('sotta', {}).get('store_features', False)))

        if feature_extractor is not None:
            extractor = feature_extractor
        elif store_features_flag:
            # only auto-select extractor when explicitly requested
            extractor = default_selector(model)
        else:
            # no extractor -> store raw input images (4D tensors) in memory
            extractor = None

        return cls(model=model,
                optimizer=optimizer,
                steps=int(steps),
                episodic=bool(episodic),
                mem_type=mem_type,
                mem_capacity=int(mem_capacity),
                hus_threshold=hus_threshold,
                hus_batch_size=int(hus_batch_size),
                esm_rho=float(esm_rho),
                num_classes=int(num_classes),
                feature_extractor=extractor)

# End of file
