#!/usr/bin/env python3
"""
Real-time progress tracker for ResNet18 learning rate grid search
Shows detailed fold-level progress and metrics
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

OUTPUT_BASE = Path("/home/lichengze/Research/DeepFeature/myresearch/phase3/phase3_outputs_ResNet18")

LR_VALUES = [
    "9.00e-05", "8.00e-05", "1.00e-04", "2.00e-04", "3.00e-04",
    "4.00e-04", "5.00e-04", "6.00e-04", "6.20e-04", "6.40e-04",
    "6.80e-04", "6.90e-04", "7.00e-04", "7.20e-04", "7.40e-04",
    "7.60e-04", "8.00e-04", "9.00e-04", "1.00e-03", "2.00e-03",
    "3.00e-03", "4.00e-03", "6.00e-03", "9.00e-03"
]

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def get_latest_epoch_info(log_file):
    """Extract latest epoch info from log file"""
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for epoch lines from the end
        for line in reversed(lines[-50:]):
            if 'Epoch' in line and 'loss' in line:
                # Parse: Epoch 023 | loss 0.1234 | Uno(int) 0.678 | ...
                match = re.search(r'Epoch\s+(\d+).*loss\s+([\d.]+).*Uno\(int\)\s+([\d.]+)', line)
                if match:
                    return {
                        'epoch': int(match.group(1)),
                        'loss': float(match.group(2)),
                        'uno_int': float(match.group(3))
                    }
    except Exception as e:
        pass
    return None

def get_checkpoint_info(fold_dir):
    """Get info from saved checkpoint"""
    ckpt_file = fold_dir / "best.pt"
    if not ckpt_file.exists():
        return None
    
    try:
        import torch
        ckpt = torch.load(ckpt_file, map_location='cpu')
        return {
            'uno_integral': ckpt.get('uno_integral', np.nan),
            'uno_24m': ckpt.get('uno_24m', np.nan),
            'harrell': ckpt.get('harrell', np.nan),
            'epoch': ckpt.get('epoch', -1)
        }
    except Exception as e:
        return None

def collect_progress():
    """Collect progress for all experiments"""
    progress = []
    
    for lr in LR_VALUES:
        exp_name = f"resnet18_lr{lr}"
        exp_dir = OUTPUT_BASE / exp_name
        
        exp_info = {
            'lr': lr,
            'status': 'Not Started',
            'completed_folds': 0,
            'folds': {}
        }
        
        if exp_dir.exists():
            for fold in range(5):
                fold_dir = exp_dir / f"fold_{fold}"
                
                # Check for various log files
                log_patterns = [
                    OUTPUT_BASE / f"{exp_name}_fold{fold}_gpu*.log",
                ]
                
                fold_info = {'status': 'Pending', 'progress': None, 'best': None}
                
                # Check if training is active
                for log_pattern in log_patterns:
                    for log_file in OUTPUT_BASE.glob(f"{exp_name}_fold{fold}_gpu*.log"):
                        latest = get_latest_epoch_info(log_file)
                        if latest:
                            fold_info['status'] = 'Training'
                            fold_info['progress'] = latest
                        break
                
                # Check if completed
                ckpt_info = get_checkpoint_info(fold_dir)
                if ckpt_info:
                    fold_info['status'] = 'Completed'
                    fold_info['best'] = ckpt_info
                    exp_info['completed_folds'] += 1
                
                exp_info['folds'][fold] = fold_info
            
            # Update experiment status
            if exp_info['completed_folds'] == 5:
                exp_info['status'] = 'Completed'
            elif exp_info['completed_folds'] > 0 or any(f['status'] == 'Training' for f in exp_info['folds'].values()):
                exp_info['status'] = 'In Progress'
        
        progress.append(exp_info)
    
    return progress

def display_progress(progress):
    """Display progress in a nice format"""
    clear_screen()
    
    print("="*100)
    print(f"  ResNet18 Learning Rate Grid Search - Progress Monitor".center(100))
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(100))
    print("="*100)
    print()
    
    # Summary statistics
    total_exps = len(LR_VALUES)
    completed_exps = sum(1 for p in progress if p['status'] == 'Completed')
    in_progress_exps = sum(1 for p in progress if p['status'] == 'In Progress')
    total_folds = total_exps * 5
    completed_folds = sum(p['completed_folds'] for p in progress)
    
    print(f"📊 Overall Progress: {completed_folds}/{total_folds} folds ({completed_folds/total_folds*100:.1f}%)")
    print(f"✅ Completed Experiments: {completed_exps}/{total_exps}")
    print(f"🔄 In Progress: {in_progress_exps}")
    print()
    
    # Detailed progress for each LR
    print("-"*100)
    print(f"{'LR':<12} {'Status':<12} {'Folds':<8} {'Fold Details':<65}")
    print("-"*100)
    
    for exp in progress:
        lr = exp['lr']
        status = exp['status']
        completed = exp['completed_folds']
        
        # Build fold status string
        fold_strs = []
        for fold in range(5):
            f_info = exp['folds'].get(fold, {})
            f_status = f_info.get('status', 'Pending')
            
            if f_status == 'Completed':
                best = f_info.get('best', {})
                uno = best.get('uno_integral', np.nan)
                fold_strs.append(f"F{fold}:✓({uno:.3f})")
            elif f_status == 'Training':
                prog = f_info.get('progress', {})
                ep = prog.get('epoch', 0)
                fold_strs.append(f"F{fold}:⏳(E{ep})")
            else:
                fold_strs.append(f"F{fold}:-")
        
        fold_details = " ".join(fold_strs)
        
        # Color coding for status
        status_icon = "✅" if status == "Completed" else "🔄" if status == "In Progress" else "⏸️"
        
        print(f"{lr:<12} {status_icon} {status:<10} {completed}/5    {fold_details}")
    
    print("-"*100)
    print()
    
    # Show currently training details
    training_now = []
    for exp in progress:
        for fold, f_info in exp['folds'].items():
            if f_info['status'] == 'Training':
                training_now.append((exp['lr'], fold, f_info['progress']))
    
    if training_now:
        print("🔥 Currently Training:")
        print()
        for lr, fold, prog in training_now:
            if prog:
                print(f"  LR {lr} | Fold {fold} | Epoch {prog['epoch']:3d} | Loss {prog['loss']:.4f} | Uno {prog['uno_int']:.3f}")
        print()
    
    # Show completed experiments summary
    completed = [p for p in progress if p['status'] == 'Completed']
    if completed:
        print("✅ Completed Experiments (with mean C-index):")
        print()
        
        results = []
        for exp in completed:
            unos = []
            for fold in range(5):
                best = exp['folds'][fold].get('best', {})
                uno = best.get('uno_integral', np.nan)
                if not np.isnan(uno):
                    unos.append(uno)
            
            if len(unos) == 5:
                mean_uno = np.mean(unos)
                std_uno = np.std(unos)
                results.append((exp['lr'], mean_uno, std_uno))
        
        # Sort by mean uno
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  {'Rank':<6} {'LR':<12} {'Mean C-index':<20}")
        print("  " + "-"*40)
        for rank, (lr, mean, std) in enumerate(results, 1):
            print(f"  {rank:<6} {lr:<12} {mean:.4f} ± {std:.4f}")
        print()
    
    print("="*100)
    print("Press Ctrl+C to exit | Refreshing every 10 seconds...")
    print("="*100)

def main():
    """Main monitoring loop"""
    try:
        while True:
            progress = collect_progress()
            display_progress(progress)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()

