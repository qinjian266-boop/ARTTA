"""
Adversarial Example Detector - Multi-domain Feature Squeezing & Fusion
---------------------------------------------------------------------
This module implements an adversarial detection pipeline based on multi-domain 
feature extraction, including Uncertainty ($S^{UN}$), Spatial-domain ($S^{SD}$), 
and Frequency-domain ($S^{FS}$) analysis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score ,f1_score, precision_score, recall_score# 导入 roc_auc_score
import math
from collections import deque
import torchvision.transforms.functional as TF
from collections import Counter 
import logging 
import io 
from PIL import Image 
import torchvision.transforms.functional as TF_pil 
import torchvision.transforms as T 
import os 
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader 
from torch.optim.lr_scheduler import StepLR 
import matplotlib.pyplot as plt

class AdvFilter:
    """Core adversarial detector using adaptive thresholding and multi-feature fusion.
    
    Attributes:
        score_window (deque): Sliding window for dynamic threshold calculation.
        weights (Tensor): Learnable or fixed weights for feature fusion.
        all_y_true (list): Ground truth labels for cumulative evaluation.
        all_y_scores (list): Fusion scores for AUROC/FPR calculation.
    """

    def __init__(self, window_size, threshold_method, quantile_val, std_factor, weights, logger, device='cpu'):
        """
        Args:
            window_size (int): Size of the historical score buffer.
            threshold_method (str): Method to define decision boundary ('std' or 'quantile').
            quantile_val (float): Quantile ratio for thresholding.
            std_factor (float): Multiplier for standard deviation based threshold.
            weights (list): Domain weights [Uncertainty, Magnitude, Spatial, Frequency].
            logger (logging.Logger): System logger for experiment tracking.
        """
        # --- Detector Configuration ---
        self.threshold_method = threshold_method
        self.quantile_val = quantile_val
        self.std_factor = std_factor
        self.weights = torch.tensor(weights, device=device)
        self.score_window = deque(maxlen=window_size)
        
        
        # --- Environment ---
        self.logger = logger
        self.device = device
        
        # --- Metrics Accumulators ---
        self.total_tp, self.total_tn, self.total_fp, self.total_fn = 0, 0, 0, 0
        self.all_fusion_scores_list = [] 
        self.all_is_adv_collected = None
        self.all_y_true = []
        self.all_y_scores = []

        # --- Distribution Analysis Buffers (for Visualization) ---
        self._pred_entropy_vals = []      # prediction entropy
        self._grad_dir_entropy_vals = []  # gradient direction entropy
        self._mean_spectrum_vals = []     # mean spectrum
        self._labels_is_adv = []          # Labels (0: Benign, 1: Adv)

    # ===================================================================
    # Feature Extraction Methods (Core Research Components)
    # ===================================================================
    def compute_prediction_entropy(self, logits):
        """
        Calculates Uncertainty Scoring ($S^{UN}$) via Shannon Entropy.
        
        High entropy indicates the model is uncertain, a common trait of 
        adversarial samples near decision boundaries.
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1) # Numerically stable
        return -(probs * log_probs).sum(dim=1)

    def compute_image_avg_grad_magnitude(self, images):
        """
        Computes Average Gradient Magnitude in the pixel space.
        
        Captures high-frequency pixel-level oscillations introduced by 
        adversarial perturbations (e.g., PGD, FGSM).
        """
        gray_images = TF.rgb_to_grayscale(images)
        grad_x = gray_images[:, :, :, 1:] - gray_images[:, :, :, :-1]
        grad_y = gray_images[:, :, 1:, :] - gray_images[:, :, :-1, :]
        grad_magnitude = torch.sqrt(grad_x[:, :, :-1, :]**2 + grad_y[:, :, :, :-1]**2 + 1e-6)
        return grad_magnitude.mean(dim=[1,2,3])

    def compute_gradient_direction_entropy(self, images, num_bins=8):
        """
        Calculates Spatial-domain Scoring ($S^{SD}$) via Gradient Orientation.
        
        Adversarial attacks often disrupt the natural structural gradient 
        flow of an image, leading to higher directional entropy.
        """
        gray_images = TF.rgb_to_grayscale(images)
        grad_x = gray_images[:, :, :, 1:] - gray_images[:, :, :, :-1] 
        grad_y = gray_images[:, :, 1:, :] - gray_images[:, :, :-1, :] 
        directions = torch.atan2(grad_y[:, :, :, :-1], grad_x[:, :, :-1, :])
        angles = (directions * 180 / math.pi + 180) % 360
        
        entropies = []
        for i in range(images.size(0)):
            hist = torch.histc(angles[i].flatten(), bins=num_bins, min=0, max=360)
            probs = hist / (hist.sum() + 1e-6)
            entropy = -(probs * torch.log2(probs + 1e-6)).sum()
            entropies.append(entropy)
        
        return torch.tensor(entropies, device=images.device)

    import torchvision.transforms.functional as TF

    def compute_mean_spectrum(self, images):
        """
        Calculates Frequency-domain Scoring ($S^{FS}$) via 2D-FFT.
        
        Detects anomalies in the power spectrum magnitude, targeting 
        perturbations that are quasi-invisible in the spatial domain.
        """
        gray = TF.rgb_to_grayscale(images).squeeze(1)

        if hasattr(torch, "fft") and hasattr(torch.fft, "fft2"):
            # Standard for PyTorch >= 1.8
            fft_images = torch.fft.fft2(gray)
            magnitude_spectrum = torch.abs(fft_images)
        else:
            # Legacy support
            fft_images = torch.rfft(gray, signal_ndim=2, onesided=False)
            magnitude_spectrum = torch.sqrt(fft_images[..., 0]**2 + fft_images[..., 1]**2)

        mean_spectrum = torch.mean(magnitude_spectrum, dim=[1, 2])
        return mean_spectrum

    # ===================================================================
    # Logic & Utility Functions
    # ===================================================================
    def _get_primary_scores_from_features(self, features_list):
        """Fuses normalized multi-domain features using weighted summation."""
        normalized_primary_list = []
        for scores in features_list:
            min_v, max_v = scores.min(), scores.max()
            normalized = (scores - min_v) / (max_v - min_v + 1e-6) if (max_v - min_v) > 1e-6 else torch.full_like(scores, 0.5)
            normalized_primary_list.append(normalized)
        
        normalized_primary = torch.stack(normalized_primary_list, dim=1)
        fusion_scores = (normalized_primary * self.weights).sum(dim=1)
        return fusion_scores

    
    def _remove_hooks(self, handles):
        for handle in handles:
            handle.remove()
    
    def compute_gradient_features(self, model, images, targets):
        """Advanced Gradient Analysis including Norm, Sim, and Diversity."""
        
        input_grads = self._compute_input_gradients(model, images, targets)
        
        # L2 Norm of input gradients
        grad_norms = torch.norm(input_grads.view(input_grads.size(0), -1), p=2, dim=1)
        
        # Directional Similarity (Cosine)
        grad_directions = F.normalize(input_grads.view(input_grads.size(0), -1), p=2, dim=1)
        cosine_sim = torch.mm(grad_directions, grad_directions.t())
        direction_scores = torch.mean(cosine_sim, dim=1)  # 平均余弦相似度
        
        # Gradient Angle Diversity
        batch_size = grad_directions.size(0)
        if batch_size > 3:  
            mask = 1.0 - torch.eye(batch_size, device=self.device)
            sim_values = cosine_sim * mask
            angles = torch.acos(torch.clamp(sim_values, -0.9999, 0.9999))
            angle_mean = torch.sum(angles, dim=1) / (batch_size - 1 + 1e-6)
            angle_var = torch.zeros_like(angle_mean)
         
            for i in range(batch_size):
                angle_diff = angles[i] - angle_mean[i]
                angle_var[i] = torch.sum(angle_diff * angle_diff * mask[i]) / (batch_size - 1 + 1e-6)

            angle_diversity = 1.0 - torch.sqrt(angle_var) / (math.pi/2)
        else:
            angle_diversity = torch.zeros(batch_size, device=self.device)
        
        # Coefficient of Variation (CV) for Gradient Consistency
        grad_magnitude = torch.sqrt(torch.sum(input_grads**2, dim=1))  # [B,H,W,C]
        grad_mag_mean = torch.mean(grad_magnitude.view(batch_size, -1), dim=1)
        grad_mag_std = torch.std(grad_magnitude.view(batch_size, -1), dim=1)
        grad_mag_cv = grad_mag_std / (grad_mag_mean + 1e-6)  
        mag_consistency = torch.clamp(grad_mag_cv / 2.0, 0, 1)  
        
        return {
            'grad_norms': grad_norms,
            'direction_scores': direction_scores,
            'angle_diversity': angle_diversity,
            'mag_consistency': mag_consistency
        }

    # ===================================================================
    # Main Detection Pipeline
    # ===================================================================
    def filter_batch(self, images, targets, is_adv, config, model_tta, model_orig, logger, batch_idx):
        """
        Execution pipeline for adversarial filtering.
        
        Workflow: Feature Extraction -> Score Fusion -> Adaptive Thresholding.
        """
        model_tta.eval()

        # Phase 1: Feature Extraction
        with torch.no_grad():
            logits_tta = model_tta(images)
        
        primary_features_raw = [
            self.compute_prediction_entropy(logits_tta),
            self.compute_image_avg_grad_magnitude(images),
            self.compute_gradient_direction_entropy(images),
            self.compute_mean_spectrum(images)
        ]
        # Phase 2: Score Fusion
        fusion_scores = self._get_primary_scores_from_features(primary_features_raw)
        
        # Data persistence for distribution plotting
        try:
            pred_ent = primary_features_raw[0].detach().cpu().numpy()
            grad_dir_ent = primary_features_raw[2].detach().cpu().numpy()
            mean_spec = primary_features_raw[3].detach().cpu().numpy()
            is_adv_cpu = is_adv.detach().cpu().numpy() if isinstance(is_adv, torch.Tensor) else np.array(is_adv)

            self._pred_entropy_vals.extend(pred_ent.tolist())
            self._grad_dir_entropy_vals.extend(grad_dir_ent.tolist())
            self._mean_spectrum_vals.extend(mean_spec.tolist())
            self._labels_is_adv.extend(is_adv_cpu.astype(int).tolist())
        except Exception as e:
            self.logger.warning(f"Telemetry recording failed: {e}")

        # Phase 3: Adaptive Thresholding via Sliding Window
        self.score_window.extend(fusion_scores.cpu().numpy())
        window_scores = torch.tensor(list(self.score_window), device=self.device)
        
        if self.threshold_method == 'std':
            primary_threshold = window_scores.mean() + self.std_factor * window_scores.std()
            benign_threshold = window_scores.mean() - self.std_factor * window_scores.std()
        else:
            primary_threshold = torch.quantile(window_scores, self.quantile_val)
            benign_quantile = 1.0 - self.quantile_val
            benign_threshold = torch.quantile(window_scores, benign_quantile)

        # Boolean Mask Assignment
        final_detected_mask = fusion_scores > primary_threshold
        high_confidence_benign_mask = fusion_scores < benign_threshold
        gray_area_mask = ~(final_detected_mask | high_confidence_benign_mask)

        # Phase 4: Performance Logging
        self.all_fusion_scores_list.extend(fusion_scores.cpu().numpy())
        if self.all_is_adv_collected is None:
            self.all_is_adv_collected = is_adv.clone()
        else:
            self.all_is_adv_collected = torch.cat([self.all_is_adv_collected, is_adv])
        
        self.update_statistics(final_detected_mask, is_adv)
        
        if batch_idx % config['adv_detection']['batch_stats']['print_freq'] == 0:
            self.print_batch_statistics(batch_idx, final_detected_mask, is_adv, fusion_scores, primary_threshold)
        
        self.all_y_true.extend(is_adv.cpu().numpy())
        self.all_y_scores.extend(fusion_scores.cpu().numpy())
        
        return fusion_scores, gray_area_mask, high_confidence_benign_mask, final_detected_mask, logits_tta

    # ===================================================================
    # Statistics & Logging Utilities
    # ===================================================================
    def update_statistics(self, detected_mask, is_adv):
        
        """Update detection statistics for confusion matrix.
        
        Args:
            detected_mask: Binary detection results (1 = adversarial).
            is_adv: Ground truth labels (1 = adversarial).
        """
        self.total_tp += ((detected_mask == 1) & (is_adv == 1)).sum().item()
        self.total_tn += ((detected_mask == 0) & (is_adv == 0)).sum().item()
        self.total_fp += ((detected_mask == 1) & (is_adv == 0)).sum().item()
        self.total_fn += ((detected_mask == 0) & (is_adv == 1)).sum().item()

    def print_batch_statistics(self, batch_idx, detected_mask, is_adv, fusion_scores, threshold):
        """Log detection performance for the current batch."""
        tp = ((detected_mask == 1) & (is_adv == 1)).sum().item()
        fp = ((detected_mask == 1) & (is_adv == 0)).sum().item()
        tn = ((detected_mask == 0) & (is_adv == 0)).sum().item()
        fn = ((detected_mask == 0) & (is_adv == 1)).sum().item()
        
        self.logger.info(f"Batch {batch_idx} Detection Stats:")
        self.logger.info(f"  - Score Range: [{fusion_scores.min():.4f}, {fusion_scores.max():.4f}], Threshold: {threshold:.4f}")
        self.logger.info(f"  - Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    def log_detection_summary(self):
        """Log overall detection performance after processing all batches."""
        self.logger.info("\n" + "="*50)
        self.logger.info("FINAL ADVERSARIAL DETECTION PERFORMANCE SUMMARY:")
        self.logger.info("="*50)
        
        total_samples = self.total_tp + self.total_fp + self.total_tn + self.total_fn
        if total_samples == 0:
            self.logger.info("No samples processed; summary unavailable.")
            return

        # 计算和打印基本指标
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (self.total_tp + self.total_tn) / total_samples
        
        self.logger.info(f"Total Samples: {total_samples}")
        self.logger.info(f"TP: {self.total_tp} | FP: {self.total_fp} | TN: {self.total_tn} | FN: {self.total_fn}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall (TPR): {recall:.4f}")
        self.logger.info(f"F1 Score: {f1_score:.4f}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")

        # Compute AUROC and FPR@TPR95
        try:
            y_true = np.array(self.all_y_true)
            y_scores = np.array(self.all_y_scores)
            
            # Require both positive and negative samples
            if len(np.unique(y_true)) > 1:
                # Calculate AUROC
                auroc_score = roc_auc_score(y_true, y_scores)
                self.logger.info(f"检测器 AUROC: {auroc_score:.4f}")

                # Calculate FPR at 95% TPR
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                
                if np.max(tpr) >= 0.95:
                    idx_tpr95 = np.where(tpr >= 0.95)[0][0]
                    fpr_at_tpr95 = fpr[idx_tpr95]
                    self.logger.info(f"Detector FPR@TPR95: {fpr_at_tpr95:.4f}")
                else:
                    self.logger.info("TPR did not reach 95%; skip FPR@TPR95.")
            else:
                self.logger.info("Single-class label; skip AUROC and FPR@TPR95.")
        except Exception as e:
            self.logger.error(f"Error computing AUROC/FPR: {e}")
        
        self.logger.info("="*50 + "\n")


    def compute_and_print_auc(self, logger):
        """Compute and log cumulative AUC score."""
        if not self.all_fusion_scores_list or self.all_is_adv_collected is None:
            logger.info("Insufficient data for AUC calculation.")
            return

        y_true = self.all_is_adv_collected.cpu().numpy()
        y_scores = np.array(self.all_fusion_scores_list)

        if len(np.unique(y_true)) < 2:
            logger.info("Single-class data; cannot compute AUC.")
            return

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        logger.info(f"Preliminary Detector Cumulative AUC: {roc_auc:.4f}")

    # ===================================================================
    # Visualization Utilities
    # ===================================================================


    def plot_feature_distributions(self, output_dir='./results',
                             dpi=600,
                             formats=('png', 'pdf'),
                             legend_fontsize=14,
                             label_fontsize=16,
                             tick_fontsize=14,
                             title_fontsize=16,
                             bins=60):
        """Plot feature distributions for paper visualization."""
        import os
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        os.makedirs(output_dir, exist_ok=True)

        pred   = np.array(self._pred_entropy_vals)
        gdir   = np.array(self._grad_dir_entropy_vals)
        mspec  = np.array(self._mean_spectrum_vals)
        labels = np.array(self._labels_is_adv, dtype=int)

        if pred.size == 0:
            self.logger.info("No feature data collected; skip plotting.")
            return {}

        benign_mask = labels == 0
        adv_mask    = labels == 1

        def compute_stats(arr_b, arr_a):
            rb = arr_b[~np.isnan(arr_b)]
            ra = arr_a[~np.isnan(arr_a)]
            mean_b = np.mean(rb) if len(rb) else np.nan
            mean_a = np.mean(ra) if len(ra) else np.nan
            std_b  = np.std(rb, ddof=0) if len(rb) else np.nan
            std_a  = np.std(ra, ddof=0) if len(ra) else np.nan
            fisher = ((mean_b - mean_a)**2) / (std_b**2 + std_a**2 + 1e-12)
            pooled = np.sqrt(0.5 * (std_b**2 + std_a**2) + 1e-12)
            cohens = abs(mean_b - mean_a) / (pooled + 1e-12)
            return dict(mean_b=mean_b, mean_a=mean_a,
                        std_b=std_b, std_a=std_a,
                        fisher=fisher, cohens_d=cohens,
                        n_b=len(rb), n_a=len(ra))

        s_pred = compute_stats(pred[benign_mask], pred[adv_mask])
        s_gdir = compute_stats(gdir[benign_mask], gdir[adv_mask])
        s_msp  = compute_stats(mspec[benign_mask], mspec[adv_mask])

        # 保存统计文本 (保持不变)
        stats_path = os.path.join(output_dir, 'feature_distribution_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("Feature distribution statistics (benign vs adversarial)\n\n")
            for name, s in [('prediction_entropy', s_pred),
                            ('grad_dir_entropy', s_gdir),
                            ('mean_spectrum', s_msp)]:
                f.write(f"{name}:\n")
                for k, v in s.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
        self.logger.info(f"Feature statistics saved to: {stats_path}")

        plt.rcParams.update({
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'font.family': 'serif',
            'axes.labelsize': label_fontsize,
            'axes.titlesize': title_fontsize,
            'xtick.labelsize': tick_fontsize,
            'ytick.labelsize': tick_fontsize,
            'legend.fontsize': legend_fontsize,
            'text.usetex': False,  
        })

        alpha    = 0.65
        edge_kws = {'edgecolor': 'black', 'linewidth': 0.5}

        def plot_panel(ax, data_b, data_a, s, xlabel, title,
                    focus_main_body=False,
                    legend_loc='upper right',
                    show_title=False):
            data_b = np.array(data_b, dtype=float)
            data_a = np.array(data_a, dtype=float)
            data_b = data_b[np.isfinite(data_b)]
            data_a = data_a[np.isfinite(data_a)]

            if data_b.size == 0:
                ax.text(0.5, 0.5, "No benign samples", transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, alpha=0.8)
                data_b = np.array([0.0])
            if data_a.size == 0:
                ax.text(0.5, 0.4, "No adversarial samples", transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, alpha=0.8)
                data_a = np.array([0.0])

            if np.isclose(data_b.max(), data_b.min()):
                data_b += np.random.normal(scale=1e-8, size=data_b.shape)
            if np.isclose(data_a.max(), data_a.min()):
                data_a += np.random.normal(scale=1e-8, size=data_a.shape)

            if focus_main_body:
                low_cut  = np.percentile(np.concatenate([data_b, data_a]), 5)
                high_cut = np.percentile(np.concatenate([data_b, data_a]), 95)
                data_b   = data_b[(data_b >= low_cut) & (data_b <= high_cut)]
                data_a   = data_a[(data_a >= low_cut) & (data_a <= high_cut)]
                ax.set_xlim(low_cut, high_cut)

            try:
                h_b = ax.hist(data_b, bins=bins, density=True, alpha=alpha,
                            label=f"Benign (n={len(data_b)})", **edge_kws)
                h_a = ax.hist(data_a, bins=bins, density=True, alpha=alpha,
                            label=f"Adversarial (n={len(data_a)})", **edge_kws)
            except Exception:
                h_b = ax.hist(data_b, bins=bins, density=False, alpha=alpha,
                            label=f"Benign (n={len(data_b)})", **edge_kws)
                h_a = ax.hist(data_a, bins=bins, density=False, alpha=alpha,
                            label=f"Adversarial (n={len(data_a)})", **edge_kws)

            ax.axvline(np.mean(data_b), color='blue', linestyle='--', linewidth=1.6, alpha=0.9)
            ax.axvline(np.mean(data_a), color='red',  linestyle='--', linewidth=1.6, alpha=0.9)

            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.set_ylabel("Probability Density", fontsize=label_fontsize)

            if show_title:
                ax.set_title(f"{title}\n(Fisher: {s['fisher']:.3f})", fontsize=title_fontsize)

            ax.grid(True, linestyle=':', linewidth=0.6)

            mean_adv_label = f"Adv Mean: {np.mean(data_a):.3f}"
            mean_ben_label = f"Benign Mean: {np.mean(data_b):.3f}"

            hist_handles = []
            if len(h_b) >= 3 and len(h_b[2]) > 0:
                hist_handles.append(h_b[2][0])
            if len(h_a) >= 3 and len(h_a[2]) > 0:
                hist_handles.append(h_a[2][0])

            line_adv = Line2D([0], [0], color='red',  linestyle='--', linewidth=1.6)
            line_ben = Line2D([0], [0], color='blue', linestyle='--', linewidth=1.6)

            handles = hist_handles + [line_adv, line_ben]
            labels  = [f"Benign (n={len(data_b)})",
                    f"Adversarial (n={len(data_a)})"] + [mean_adv_label, mean_ben_label]

            leg = ax.legend(handles=handles, labels=labels, loc=legend_loc,
                            frameon=True, fancybox=True, framealpha=0.9,
                            edgecolor='black', fontsize=legend_fontsize)
            leg.get_frame().set_linewidth(0.8)

        saved_paths = {}
        fig_size = (8, 6)

        fig_pred, ax_pred = plt.subplots(1, 1, figsize=fig_size)
        plot_panel(ax_pred, pred[benign_mask], pred[adv_mask], s_pred,
                r"Uncertainty Scoring (Prediction Entropy, $S^{UN}$)", "Uncertainty Scoring",
                legend_loc='upper right',
                show_title=False)
        plt.tight_layout(pad=0.1)
        base_name_pred = 'feature_dist_pred_entropy'
        for fmt in formats:
            try:
                out_path = os.path.join(output_dir, f"{base_name_pred}.{fmt}")
                fig_pred.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
                saved_paths[f'pred_entropy_{fmt}'] = out_path
            except Exception as e:
                self.logger.error(f"保存 {out_path} 失败: {e}")
        plt.close(fig_pred)


        fig_gdir, ax_gdir = plt.subplots(1, 1, figsize=fig_size)
        plot_panel(ax_gdir, gdir[benign_mask], gdir[adv_mask], s_gdir,
                r"Spatial-domain Scoring (Gradient Direction Entropy, $S^{SD}$)", "Spatial-domain Scoring",
                legend_loc='upper left',
                show_title=False,
                focus_main_body=True)
        plt.tight_layout(pad=0.1)
        base_name_gdir = 'feature_dist_grad_dir_entropy'
        for fmt in formats:
            try:
                out_path = os.path.join(output_dir, f"{base_name_gdir}.{fmt}")
                fig_gdir.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
                saved_paths[f'grad_dir_entropy_{fmt}'] = out_path
            except Exception as e:
                self.logger.error(f"保存 {out_path} 失败: {e}")
        plt.close(fig_gdir)

        fig_msp, ax_msp = plt.subplots(1, 1, figsize=fig_size)
        plot_panel(ax_msp, mspec[benign_mask], mspec[adv_mask], s_msp,
                r"Frequency Spectrum Scoring (Spectrum Mean Value, $S^{FS}$)", "Frequency Spectrum Scoring",
                legend_loc='upper left',
                show_title=False)
        plt.tight_layout(pad=0.1)
        base_name_msp = 'feature_dist_mean_spectrum'
        for fmt in formats:
            try:
                out_path = os.path.join(output_dir, f"{base_name_msp}.{fmt}")
                fig_msp.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
                saved_paths[f'mean_spectrum_{fmt}'] = out_path
            except Exception as e:
                self.logger.error(f"Failed to save {out_path}: {e}")
        plt.close(fig_msp)

        self.logger.info(f"All feature distribution plots saved to: {output_dir}")
        return saved_paths
    
    

# 简单使用示例
if __name__ == "__main__":
    """Minimal test for AdvFilter."""
    test_logger = logging.getLogger(__name__ + "_test_adv_filter") # Changed name to be more specific
    test_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not test_logger.hasHandlers(): 
        test_logger.addHandler(handler)
    test_logger.propagate = False


    images = torch.randn(20, 3, 32, 32) 
    targets = torch.randint(0, 10, (20,))
    is_adv = torch.zeros(20, dtype=torch.bool)
    is_adv[::2] = True
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 模拟一个简化的ResNet结构，以便于现有代码运行
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2) # 32 -> 16
            self.layer1 = nn.Identity()
            self.layer2 = nn.Identity()
            self.layer3 = nn.Identity()
            self.layer4 = nn.Identity()
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            # 定义一个完整的forward，使其可以被挤压分数函数直接调用
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    dummy_model = DummyModel()
    
    dummy_config = {
        'adv_detection': {
            'batch_stats': {'print_freq': 1},
        }
    }


    filter_instance = AdvFilter(
        logger=test_logger, 
        window_size=20, 
        threshold_method='std', 
        quantile_val=0.8, 
        std_factor=1.0, 
        weights=[0.25, 0.25, 0.25, 0.25],
    )

    
    dataset = TensorDataset(images, targets, is_adv)
    loader = DataLoader(dataset, batch_size=10)

    for batch_idx, (batch_images, batch_targets, batch_is_adv) in enumerate(loader):
        filtered_images, mask, scores, final_detected_as_adv = filter_instance.filter_batch(
            batch_images, batch_targets, batch_is_adv, 
            dummy_config, dummy_model, dummy_model, test_logger, batch_idx
        )
        test_logger.info(f"Batch {batch_idx+1} processed.")
    
    filter_instance.log_detection_summary()

