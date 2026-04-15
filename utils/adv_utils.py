import os
import os.path as osp
import torch
import numpy as np

def build_filtered_dataloader(original_data_loader, model, adv_detector, dataset_name):
    """构建过滤后的数据加载器"""
    
    # 如果不需要检测或过滤，直接返回原始数据加载器
    if not adv_detector.enable_detection or not adv_detector.enable_filtering:
        return original_data_loader
    
    # 收集所有过滤后的数据
    filtered_data = []
    dataset = original_data_loader.dataset
    
    print(f"开始对数据集 '{dataset_name}' 进行对抗样本检测过滤...")
    
    # 获取样本记录（用于知道哪些是对抗样本）
    sample_records = []
    if hasattr(dataset, 'get_sample_records'):
        sample_records = dataset.get_sample_records()
    else:
        # 如果没有样本记录，创建一个默认的（所有样本都标记为正常）
        sample_records = [{'filename': f'sample_{i}', 'is_adv': False} 
                         for i in range(len(dataset))]
    
    # 遍历原始数据加载器
    for batch_idx, data in enumerate(original_data_loader):
        images = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        
        # 准备样本信息用于检测器记录
        image_filenames = []
        is_adv_labels = []
        
        # 获取当前批次的样本信息
        start_idx = batch_idx * original_data_loader.batch_size
        end_idx = min(start_idx + original_data_loader.batch_size, len(sample_records))
        for j in range(start_idx, end_idx):
            if j < len(sample_records):
                image_filenames.append(sample_records[j]['filename'])
                is_adv_labels.append(1 if sample_records[j]['is_adv'] else 0)
        
        # 检测并过滤对抗样本
        detected_mask = adv_detector.detect_adv_samples(
            images, model, image_filenames, is_adv_labels
        )
        
        # 过滤掉对抗样本，只保留干净样本
        clean_mask = ~detected_mask
        if clean_mask.any():
            filtered_images = images[clean_mask]
            filtered_img_metas = [img_metas[i] for i in range(len(img_metas)) if clean_mask[i]]
            
            # 创建过滤后的数据批次
            filtered_batch = {
                'img': [filtered_images],
                'img_metas': [filtered_img_metas]
            }
            
            # 如果原始数据中有其他键，也保留它们
            for key in data.keys():
                if key not in ['img', 'img_metas']:
                    filtered_batch[key] = data[key]
            
            filtered_data.append(filtered_batch)
            
            print(f"批次 {batch_idx}: 检测到 {detected_mask.sum().item()}/{len(detected_mask)} 个对抗样本，过滤后剩余 {filtered_images.size(0)} 个干净样本")
        else:
            print(f"批次 {batch_idx}: 所有样本都被检测为对抗样本，跳过该批次")
    
    print(f"过滤完成: 原始批次 {len(original_data_loader)}，过滤后批次 {len(filtered_data)}")
    
    # 创建自定义的数据加载器返回过滤后的数据
    class FilteredDataLoader:
        def __init__(self, data):
            self.data = data
            self.batch_size = 1  # 因为我们是一个批次一个批次处理的
            self.dataset = original_data_loader.dataset  # 保留原始数据集引用
        
        def __iter__(self):
            return iter(self.data)
        
        def __len__(self):
            return len(self.data)
    
    return FilteredDataLoader(filtered_data)


def update_adv_paths_in_config(cfg, args):
    """更新配置中的对抗样本路径"""
    if args.adv_data_root:
        # 更新所有测试集的对抗样本路径
        test_datasets = ['test', 'test1', 'test2', 'test3', 'val']
        for dataset_name in test_datasets:
            if hasattr(cfg.data, dataset_name):
                dataset_cfg = getattr(cfg.data, dataset_name)
                if hasattr(dataset_cfg, 'adv_img_dir') and args.adv_img_dir:
                    # 构建完整的对抗样本路径
                    dataset_cfg.adv_img_dir = args.adv_img_dir
                    print(f"更新 {dataset_name} 的对抗样本路径: {dataset_cfg.adv_img_dir}")