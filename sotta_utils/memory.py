# sotta_utils/memory.py
"""
Lightweight memory implementations (FIFO, HUS, ConfFIFO) adapted so they do NOT
depend on a global `conf` module at import time. Instead they accept
parameters (num_class, device...) in constructors when needed.

This file is intended to be used within the sotta_utils package (relative import).
It keeps the original repo API (add_instance, get_memory, save_state_dict, set_memory, ...).
"""

import random
import numpy as np
import torch

# Device fallback: do not rely on conf; pick first available CUDA or cpu.
_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FIFO:
    def __init__(self, capacity, device=None):
        self.data = [[], [], []]  # feats, cls, domain
        self.capacity = int(capacity)
        self.device = _device if device is None else device

    def set_memory(self, state_dict):
        self.data = [ls[:] for ls in state_dict['data']]
        if 'capacity' in state_dict:
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        return {'data': [ls[:] for ls in self.data], 'capacity': self.capacity}

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 3)
        if self.get_occupancy() >= self.capacity:
            self.remove_instance()
        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)


class HUS:
    def __init__(self, capacity, threshold=None, num_class: int = 1000, device=None):
        """
        HUS memory adapted: accept num_class explicitly (default 1000).
        """
        self.num_class = int(num_class)
        self.data = [[[], [], [], []] for _ in range(self.num_class)]  # feat, pseudo_cls, domain, conf
        self.counter = [0] * self.num_class
        self.marker = [''] * self.num_class
        self.capacity = int(capacity)
        self.threshold = threshold
        self.device = _device if device is None else device

    def set_memory(self, state_dict):
        self.data = [[l[:] for l in ls] for ls in state_dict['data']]
        self.counter = state_dict.get('counter', self.counter)[:]
        self.marker = state_dict.get('marker', self.marker)[:]
        self.capacity = state_dict.get('capacity', self.capacity)
        self.threshold = state_dict.get('threshold', self.threshold)

    def save_state_dict(self):
        return {
            'data': [[l[:] for l in ls] for ls in self.data],
            'counter': self.counter[:],
            'marker': self.marker[:],
            'capacity': self.capacity,
            'threshold': self.threshold
        }

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):
        tmp_data = [[], [], []]
        for data_per_cls in self.data:
            feats, cls, dls, _ = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)
        return tmp_data

    def get_occupancy(self):
        return sum(len(d[0]) for d in self.data)

    def get_occupancy_per_class(self):
        return [len(data_per_cls[0]) for data_per_cls in self.data]

    def add_instance(self, instance):
        # expects (feat, cls, domain, conf)
        assert (len(instance) == 4)
        cls = int(instance[1])
        if cls < 0 or cls >= self.num_class:
            # clamp to valid range
            cls = max(0, min(self.num_class - 1, cls))
        self.counter[cls] += 1
        is_add = True
        if self.threshold is not None and instance[3] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class) if occupancy_per_class else 0
        return [i for i, oc in enumerate(occupancy_per_class) if oc == max_value]

    def get_average_confidence(self):
        conf_list = []
        for data_per_cls in self.data:
            conf_list.extend(data_per_cls[3])
        return float(np.average(conf_list)) if conf_list else 0.0

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices and largest_indices:
            largest = random.choice(largest_indices)
            tgt_idx = self.get_target_index(self.data[largest][3])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:
            tgt_idx = self.get_target_index(self.data[cls][3]) if len(self.data[cls][3]) > 0 else None
            if tgt_idx is not None:
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
        return True

    def reset_value(self, feats, cls, aux):
        self.data = [[[], [], [], []] for _ in range(self.num_class)]
        for i in range(len(feats)):
            tgt_idx = int(cls[i])
            if tgt_idx < 0 or tgt_idx >= self.num_class:
                continue
            self.data[tgt_idx][0].append(feats[i])
            self.data[tgt_idx][1].append(cls[i])
            self.data[tgt_idx][2].append(0)
            self.data[tgt_idx][3].append(aux[i])


class ConfFIFO:
    def __init__(self, capacity, threshold=0.0, device=None):
        self.data = [[], [], [], []]
        self.capacity = int(capacity)
        self.threshold = threshold
        self.device = _device if device is None else device

    def set_memory(self, state_dict):
        self.data = [ls[:] for ls in state_dict['data']]
        self.threshold = state_dict.get('threshold', self.threshold)
        if 'capacity' in state_dict:
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        return {'data': [ls[:] for ls in self.data], 'capacity': self.capacity, 'threshold': self.threshold}

    def get_memory(self):
        return self.data[:3]

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        # expects (feat, cls, domain, conf)
        assert (len(instance) == 4)
        if instance[3] < self.threshold:
            return
        if self.get_occupancy() >= self.capacity:
            self.remove_instance()
        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)

    def reset_value(self, feats, cls, aux):
        self.data = [[], [], [], []]
