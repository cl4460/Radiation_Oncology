#!/usr/bin/env python
"""
test_single_model.py - 验证单个模型的C-index
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_ipcw

# 从您的代码导入必要的函数和类
# ... (导入所有必要的代码) ...

def test_single_model(fold, lr, manifest, idx_train, idx_val):
    """
    测试单个模型能否复现checkpoint中的C-index
    """
    print(f"\n{'='*70}")
    print(f"Testing: Fold {fold}, LR={lr}")
    print(f"{'='*70}")
    
    # 加载模型
    model_path = Path(f'.../lr_{lr}/fold_{fold}/best.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    
    print(f"\nCheckpoint info:")
    print(f"  Saved uno_integral: {ckpt.get('uno_integral', 'N/A')}")
    print(f"  Saved uno_24m: {ckpt.get('uno_24m', 'N/A')}")
    print(f"  Use TTA: {ckpt.get('use_tta', 'N/A')}")
    
    # 重建模型
    model = ResNet10_Clinical_GN(
        in_channels=2,
        clinical_dim=ckpt['clinical_dim'],
        n_bins=ckpt['n_time_bins'],
        dropout=0.35
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to('cuda:0')
    
    # 准备数据（与ensemble代码相同）
    times = manifest["time"].values
    events = manifest["event"].values
    cuts = np.array(ckpt['cuts'])
    n_time_bins = ckpt['n_time_bins']
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
    
    # ... 准备dataloader（与ensemble代码相同）...
    
    # IPCW
    y_train_struct = to_structured_y(times[idx_train], events[idx_train])
    y_val_struct = to_structured_y(times[idx_val], events[idx_val])
    evt_times = times[idx_train][events[idx_train] == 1]
    tau = np.quantile(evt_times, 0.9)
    
    # 预测
    all_surv = []
    for batch in tqdm(dl_val, desc="Predicting"):
        # 测试1: 不用TTA
        model.eval()
        with torch.no_grad():
            logits = model(batch['image'].to('cuda:0'), batch['clinical'].to('cuda:0'))
            haz = torch.sigmoid(logits[:, :n_time_bins])
            surv = torch.cumprod(1.0 - haz, dim=1)
        all_surv.append(surv.cpu().numpy())
    
    all_surv = np.concatenate(all_surv, axis=0)
    risk = -np.trapz(all_surv, x=bin_mids, axis=1)
    
    uno_no_tta = concordance_index_ipcw(y_train_struct, y_val_struct, risk, tau=tau)[0]
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Checkpoint uno: {ckpt.get('uno_integral', 'N/A')}")
    print(f"  Reproduced (no TTA): {uno_no_tta:.4f}")
    print(f"  Difference: {abs(uno_no_tta - ckpt.get('uno_integral', 0)):.4f}")
    print(f"{'='*70}")
    
    # 测试2: 用TTA
    print(f"\nTesting with TTA...")
    all_surv_tta = []
    for batch in tqdm(dl_val, desc="Predicting (TTA)"):
        surv = predict_with_tta4(model, batch['image'], batch['clinical'], 
                                'cuda:0', n_time_bins)
        all_surv_tta.append(surv)
    
    all_surv_tta = np.concatenate(all_surv_tta, axis=0)
    risk_tta = -np.trapz(all_surv_tta, x=bin_mids, axis=1)
    uno_tta = concordance_index_ipcw(y_train_struct, y_val_struct, risk_tta, tau=tau)[0]
    
    print(f"\n  With TTA: {uno_tta:.4f}")
    print(f"  Difference: {abs(uno_tta - ckpt.get('uno_integral', 0)):.4f}")

# 运行测试
# test_single_model(fold=0, lr='7e-4', manifest, idx_train, idx_val)