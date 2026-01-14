#!/usr/bin/env python3
"""
ACE Visualization Tool
======================
Consolidated visualization script for ACE experiment analysis.

Usage:
    python visualize.py <run_dir>              # Generate all visualizations
    python visualize.py <run_dir> --dashboard  # Only success dashboard
    python visualize.py --demo                 # Demo mode with untrained model

Outputs:
    - success_verification.png  : Dashboard with success criteria
    - mechanism_contrast.png    : Learned vs ground truth mechanisms (if model available)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Optional torch imports for mechanism visualization
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# DATA LOADING
# =============================================================================

def load_run_data(run_dir):
    """Load all CSV data from a run directory."""
    data = {}
    
    files = {
        'node_losses': 'node_losses.csv',
        'metrics': 'metrics.csv', 
        'value_diversity': 'value_diversity.csv',
        'dpo': 'dpo_training.csv'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(run_dir, filename)
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath)
    
    return data


# =============================================================================
# SUCCESS DASHBOARD
# =============================================================================

def create_success_dashboard(data, run_dir, output_path=None):
    """
    Create comprehensive dashboard showing run success metrics.
    
    Panels:
        1. Node losses over time (log scale)
        2. Final loss bar chart with pass/fail coloring
        3. Intervention target distribution pie chart
        4. X2 value diversity scatter plot
        5. X2 value histogram
        6. X3 collider learning progress
        7. Success criteria checklist
    """
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'ACE Run Analysis: {os.path.basename(run_dir)}', fontsize=14, fontweight='bold')
    
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # Panel 1: Node Losses Over Time
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2])
    
    if 'node_losses' in data:
        nl = data['node_losses']
        episode_losses = nl.groupby('episode').last().reset_index()
        
        colors = {'loss_X1': '#2ecc71', 'loss_X2': '#3498db', 'loss_X3': '#e74c3c', 
                  'loss_X4': '#9b59b6', 'loss_X5': '#f39c12'}
        labels = {'loss_X1': 'X1 (root)', 'loss_X2': 'X2 (X1→X2)', 'loss_X3': 'X3 (COLLIDER)', 
                  'loss_X4': 'X4 (root)', 'loss_X5': 'X5 (X4→X5)'}
        
        for col in ['loss_X1', 'loss_X2', 'loss_X3', 'loss_X4', 'loss_X5']:
            if col in episode_losses.columns:
                ax1.plot(episode_losses['episode'], episode_losses[col], 
                        label=labels[col], color=colors[col], linewidth=2, alpha=0.8)
        
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='X3 Target (0.5)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Mechanism Loss')
        ax1.set_title('Node Mechanism Losses Over Training')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel 2: Final Loss Summary
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    
    if 'node_losses' in data:
        nl = data['node_losses']
        final_losses = nl.iloc[-1]
        
        nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
        losses = [final_losses.get(f'loss_{n}', 0) for n in nodes]
        
        bar_colors = []
        for n, l in zip(nodes, losses):
            if n == 'X3':
                bar_colors.append('#27ae60' if l < 0.5 else '#e74c3c')
            elif n == 'X2':
                bar_colors.append('#27ae60' if l < 0.5 else '#e74c3c')
            elif n in ['X1', 'X4']:
                bar_colors.append('#3498db')
            else:
                bar_colors.append('#27ae60' if l < 0.5 else '#f39c12')
        
        bars = ax2.bar(nodes, losses, color=bar_colors, edgecolor='black', linewidth=1.2)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Target')
        
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            ax2.annotate(f'{loss:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('Final Loss')
        ax2.set_title('Final Mechanism Losses')
        ax2.set_yscale('log')
        
        success_patch = mpatches.Patch(color='#27ae60', label='Success (< 0.5)')
        fail_patch = mpatches.Patch(color='#e74c3c', label='Failure (≥ 0.5)')
        ax2.legend(handles=[success_patch, fail_patch], loc='upper right', fontsize=8)
    
    # -------------------------------------------------------------------------
    # Panel 3: Intervention Distribution
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    
    if 'metrics' in data:
        metrics = data['metrics']
        target_counts = metrics['target'].value_counts()
        
        colors_pie = {'X1': '#2ecc71', 'X2': '#3498db', 'X3': '#e74c3c', 
                      'X4': '#9b59b6', 'X5': '#f39c12'}
        pie_colors = [colors_pie.get(t, '#95a5a6') for t in target_counts.index]
        
        wedges, texts, autotexts = ax3.pie(target_counts.values, labels=target_counts.index, 
                                            autopct='%1.1f%%', colors=pie_colors,
                                            explode=[0.05 if t == 'X2' else 0 for t in target_counts.index])
        ax3.set_title('Intervention Target Distribution')
        
        for autotext in autotexts:
            autotext.set_fontweight('bold')
    
    # -------------------------------------------------------------------------
    # Panel 4: X2 Value Diversity Over Time
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    
    if 'value_diversity' in data:
        vd = data['value_diversity']
        x2_data = vd[vd['node'] == 'X2']
        
        if len(x2_data) > 0:
            scatter = ax4.scatter(x2_data['episode'], x2_data['value'], 
                                  alpha=0.3, s=10, c=x2_data['episode'], cmap='viridis')
            ax4.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Trap value (1.5)')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('X2 Intervention Value')
            ax4.set_title('X2 Value Diversity Over Time')
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel 5: X2 Value Histogram
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    
    if 'value_diversity' in data:
        vd = data['value_diversity']
        x2_data = vd[vd['node'] == 'X2']
        
        if len(x2_data) > 0:
            ax5.hist(x2_data['value'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
            ax5.axvline(x=1.5, color='red', linestyle='--', linewidth=2, label='Trap value (1.5)')
            ax5.set_xlabel('X2 Value')
            ax5.set_ylabel('Frequency')
            ax5.set_title('X2 Intervention Value Distribution')
            ax5.legend(loc='upper right', fontsize=8)
    
    # -------------------------------------------------------------------------
    # Panel 6: X3 Collider Learning Progress
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 0])
    
    if 'node_losses' in data:
        nl = data['node_losses']
        episode_losses = nl.groupby('episode').last().reset_index()
        
        if 'loss_X3' in episode_losses.columns:
            ax6.fill_between(episode_losses['episode'], episode_losses['loss_X3'], 
                            alpha=0.3, color='#e74c3c')
            ax6.plot(episode_losses['episode'], episode_losses['loss_X3'], 
                    color='#e74c3c', linewidth=2, label='X3 Loss')
            ax6.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Target (0.5)')
            ax6.fill_between(episode_losses['episode'], 0, 0.5, alpha=0.1, color='green')
            
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('X3 Mechanism Loss')
            ax6.set_title('X3 Collider Learning Progress')
            ax6.legend(loc='upper right', fontsize=8)
            ax6.grid(True, alpha=0.3)
            
            final_x3 = episode_losses['loss_X3'].iloc[-1]
            ax6.annotate(f'Final: {final_x3:.3f}', 
                        xy=(episode_losses['episode'].iloc[-1], final_x3),
                        xytext=(-50, 20), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black'),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # -------------------------------------------------------------------------
    # Panel 7: Success Criteria Summary
    # -------------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    success_metrics = []
    
    if 'node_losses' in data:
        nl = data['node_losses']
        final = nl.iloc[-1]
        
        x3_loss = final.get('loss_X3', float('inf'))
        x2_loss = final.get('loss_X2', float('inf'))
        
        success_metrics.append(('X3 Collider Learned (< 0.5)', x3_loss < 0.5, f'{x3_loss:.3f}'))
        success_metrics.append(('X2 Mechanism Preserved (< 1.0)', x2_loss < 1.0, f'{x2_loss:.3f}'))
    
    if 'metrics' in data:
        metrics = data['metrics']
        target_counts = metrics['target'].value_counts()
        total = target_counts.sum()
        
        x2_pct = target_counts.get('X2', 0) / total * 100
        
        success_metrics.append(('X2 Interventions > 20%', x2_pct > 20, f'{x2_pct:.1f}%'))
        success_metrics.append(('Intervention Balance (X2 < 80%)', x2_pct < 80, f'{x2_pct:.1f}%'))
    
    if 'value_diversity' in data:
        vd = data['value_diversity']
        x2_vals = vd[vd['node'] == 'X2']['value']
        if len(x2_vals) > 0:
            std = x2_vals.std()
            success_metrics.append(('X2 Value Diversity (std > 1.0)', std > 1.0, f'σ={std:.2f}'))
    
    y_pos = 0.9
    ax7.text(0.5, 0.98, 'SUCCESS CRITERIA SUMMARY', fontsize=14, fontweight='bold',
             ha='center', va='top', transform=ax7.transAxes)
    
    for criterion, passed, value in success_metrics:
        color = '#27ae60' if passed else '#e74c3c'
        symbol = '✓' if passed else '✗'
        
        ax7.text(0.1, y_pos, f'{symbol}', fontsize=16, fontweight='bold', color=color,
                 ha='left', va='center', transform=ax7.transAxes)
        ax7.text(0.15, y_pos, criterion, fontsize=11,
                 ha='left', va='center', transform=ax7.transAxes)
        ax7.text(0.75, y_pos, value, fontsize=11, fontweight='bold', color=color,
                 ha='left', va='center', transform=ax7.transAxes)
        
        y_pos -= 0.15
    
    all_passed = all(passed for _, passed, _ in success_metrics)
    critical_passed = success_metrics[0][1] if success_metrics else False
    
    verdict_color = '#27ae60' if all_passed else ('#f39c12' if critical_passed else '#e74c3c')
    verdict_text = 'ALL CRITERIA MET' if all_passed else ('PRIMARY GOAL MET (with issues)' if critical_passed else 'FAILED')
    
    ax7.text(0.5, 0.1, verdict_text, fontsize=16, fontweight='bold', color=verdict_color,
             ha='center', va='center', transform=ax7.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=verdict_color, alpha=0.2))
    
    # Save
    if output_path is None:
        output_path = os.path.join(run_dir, "success_verification.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved dashboard to: {output_path}")
    return output_path


# =============================================================================
# MECHANISM CONTRAST (requires torch and model)
# =============================================================================

def create_mechanism_contrast(oracle, student, results_dir, filename="mechanism_contrast_improved.png"):
    """
    Create improved mechanism contrast visualization with:
    - Proper legends
    - Clear axis labels
    - 2D heatmaps for collider nodes (X3)
    - Error quantification
    """
    if not TORCH_AVAILABLE:
        print("Warning: torch not available, skipping mechanism contrast")
        return None
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Mechanism Comparison: Ground Truth vs Learned Student', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35,
                          left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    x_range = torch.linspace(-4, 4, 100)
    
    # Row 1: Root Distributions
    ax_x1 = fig.add_subplot(gs[0, 0])
    true_x1 = oracle.generate(2000)['X1'].detach().numpy()
    with torch.no_grad():
        pred_x1 = student.forward(2000)['X1'].detach().numpy()
    
    ax_x1.hist(true_x1, bins=40, density=True, alpha=0.5, color='black', label='Ground Truth')
    ax_x1.hist(pred_x1, bins=40, density=True, alpha=0.5, color='red', label='Student')
    ax_x1.set_xlabel('X1 Value')
    ax_x1.set_ylabel('Density')
    ax_x1.set_title('X1 Root Distribution\n(X1 ~ N(0,1))', fontsize=11)
    ax_x1.legend(loc='upper right', fontsize=8)
    
    ax_x4 = fig.add_subplot(gs[0, 1])
    true_x4 = oracle.generate(2000)['X4'].detach().numpy()
    with torch.no_grad():
        pred_x4 = student.forward(2000)['X4'].detach().numpy()
    
    ax_x4.hist(true_x4, bins=40, density=True, alpha=0.5, color='black', label='Ground Truth')
    ax_x4.hist(pred_x4, bins=40, density=True, alpha=0.5, color='red', label='Student')
    ax_x4.set_xlabel('X4 Value')
    ax_x4.set_ylabel('Density')
    ax_x4.set_title('X4 Root Distribution\n(X4 ~ N(2,1))', fontsize=11)
    ax_x4.legend(loc='upper right', fontsize=8)
    
    # Row 1: Simple Mechanisms
    ax_x2 = fig.add_subplot(gs[0, 2])
    y_true_x2 = oracle.mechanisms({'X1': x_range}, 'X2').detach().numpy()
    with torch.no_grad():
        p_tensor = x_range.unsqueeze(1)
        y_pred_x2 = student.mechanisms['X2'](p_tensor).squeeze().detach().numpy()
    
    ax_x2.plot(x_range.numpy(), y_true_x2, 'k--', lw=3, label='Truth: 2·X1 + 1')
    ax_x2.plot(x_range.numpy(), y_pred_x2, 'r-', lw=2, alpha=0.8, label='Student')
    ax_x2.set_xlabel('X1')
    ax_x2.set_ylabel('X2')
    ax_x2.set_title('X1 → X2 Mechanism\n(X2 = 2·X1 + 1)', fontsize=11)
    ax_x2.legend(loc='upper left', fontsize=8)
    ax_x2.grid(True, alpha=0.3)
    
    mse_x2 = np.mean((y_true_x2 - y_pred_x2)**2)
    ax_x2.text(0.95, 0.05, f'MSE: {mse_x2:.4f}', transform=ax_x2.transAxes,
               ha='right', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='yellow' if mse_x2 < 0.5 else 'red', alpha=0.5))
    
    ax_x5 = fig.add_subplot(gs[0, 3])
    y_true_x5 = oracle.mechanisms({'X4': x_range}, 'X5').detach().numpy()
    with torch.no_grad():
        p_tensor = x_range.unsqueeze(1)
        y_pred_x5 = student.mechanisms['X5'](p_tensor).squeeze().detach().numpy()
    
    ax_x5.plot(x_range.numpy(), y_true_x5, 'k--', lw=3, label='Truth: 0.2·X4²')
    ax_x5.plot(x_range.numpy(), y_pred_x5, 'r-', lw=2, alpha=0.8, label='Student')
    ax_x5.set_xlabel('X4')
    ax_x5.set_ylabel('X5')
    ax_x5.set_title('X4 → X5 Mechanism\n(X5 = 0.2·X4²)', fontsize=11)
    ax_x5.legend(loc='upper left', fontsize=8)
    ax_x5.grid(True, alpha=0.3)
    
    mse_x5 = np.mean((y_true_x5 - y_pred_x5)**2)
    ax_x5.text(0.95, 0.05, f'MSE: {mse_x5:.4f}', transform=ax_x5.transAxes,
               ha='right', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='yellow' if mse_x5 < 0.5 else 'red', alpha=0.5))
    
    # Row 2: X3 Collider Slices
    parents_list = oracle.get_parents('X3')
    
    ax_x3_slice1 = fig.add_subplot(gs[1, 0])
    y_true_x3_x1 = oracle.mechanisms({'X1': x_range, 'X2': torch.zeros(100)}, 'X3').detach().numpy()
    with torch.no_grad():
        p_dict = {'X1': x_range, 'X2': torch.zeros(100)}
        p_tensor = torch.stack([p_dict[p] for p in parents_list], dim=1)
        y_pred_x3_x1 = student.mechanisms['X3'](p_tensor).squeeze().detach().numpy()
    
    ax_x3_slice1.plot(x_range.numpy(), y_true_x3_x1, 'k--', lw=3, label='Truth')
    ax_x3_slice1.plot(x_range.numpy(), y_pred_x3_x1, 'r-', lw=2, alpha=0.8, label='Student')
    ax_x3_slice1.set_xlabel('X1')
    ax_x3_slice1.set_ylabel('X3')
    ax_x3_slice1.set_title('X3 vs X1 (X2=0 fixed)\n[Slice view]', fontsize=11)
    ax_x3_slice1.legend(loc='upper right', fontsize=8)
    ax_x3_slice1.grid(True, alpha=0.3)
    
    ax_x3_slice2 = fig.add_subplot(gs[1, 1])
    y_true_x3_x2 = oracle.mechanisms({'X1': torch.zeros(100), 'X2': x_range}, 'X3').detach().numpy()
    with torch.no_grad():
        p_dict = {'X1': torch.zeros(100), 'X2': x_range}
        p_tensor = torch.stack([p_dict[p] for p in parents_list], dim=1)
        y_pred_x3_x2 = student.mechanisms['X3'](p_tensor).squeeze().detach().numpy()
    
    ax_x3_slice2.plot(x_range.numpy(), y_true_x3_x2, 'k--', lw=3, label='Truth')
    ax_x3_slice2.plot(x_range.numpy(), y_pred_x3_x2, 'r-', lw=2, alpha=0.8, label='Student')
    ax_x3_slice2.set_xlabel('X2')
    ax_x3_slice2.set_ylabel('X3')
    ax_x3_slice2.set_title('X3 vs X2 (X1=0 fixed)\n[Slice view]', fontsize=11)
    ax_x3_slice2.legend(loc='upper right', fontsize=8)
    ax_x3_slice2.grid(True, alpha=0.3)
    
    # Row 2: X3 2D Heatmaps
    x1_grid = torch.linspace(-3, 3, 50)
    x2_grid = torch.linspace(-3, 3, 50)
    X1_mesh, X2_mesh = torch.meshgrid(x1_grid, x2_grid, indexing='ij')
    X1_flat = X1_mesh.flatten()
    X2_flat = X2_mesh.flatten()
    
    X3_true = oracle.mechanisms({'X1': X1_flat, 'X2': X2_flat}, 'X3').detach().numpy().reshape(50, 50)
    
    with torch.no_grad():
        p_dict = {'X1': X1_flat, 'X2': X2_flat}
        p_tensor = torch.stack([p_dict[p] for p in parents_list], dim=1)
        X3_pred = student.mechanisms['X3'](p_tensor).squeeze().detach().numpy().reshape(50, 50)
    
    ax_heat_true = fig.add_subplot(gs[1, 2])
    im1 = ax_heat_true.imshow(X3_true.T, extent=[-3, 3, -3, 3], origin='lower', 
                               aspect='auto', cmap='viridis')
    ax_heat_true.set_xlabel('X1')
    ax_heat_true.set_ylabel('X2')
    ax_heat_true.set_title('Ground Truth X3\n(X3 = 0.5·X1 - X2 + sin(X2))', fontsize=11)
    plt.colorbar(im1, ax=ax_heat_true, label='X3')
    
    ax_heat_pred = fig.add_subplot(gs[1, 3])
    im2 = ax_heat_pred.imshow(X3_pred.T, extent=[-3, 3, -3, 3], origin='lower',
                               aspect='auto', cmap='viridis')
    ax_heat_pred.set_xlabel('X1')
    ax_heat_pred.set_ylabel('X2')
    ax_heat_pred.set_title('Student Learned X3\n(Neural Network)', fontsize=11)
    plt.colorbar(im2, ax=ax_heat_pred, label='X3')
    
    # Row 3: Error Analysis
    ax_error = fig.add_subplot(gs[2, 0:2])
    X3_error = np.abs(X3_true - X3_pred)
    im3 = ax_error.imshow(X3_error.T, extent=[-3, 3, -3, 3], origin='lower',
                          aspect='auto', cmap='Reds')
    ax_error.set_xlabel('X1')
    ax_error.set_ylabel('X2')
    ax_error.set_title(f'X3 Absolute Error |Truth - Student|\nMean: {X3_error.mean():.4f}, Max: {X3_error.max():.4f}', 
                       fontsize=11)
    plt.colorbar(im3, ax=ax_error, label='|Error|')
    ax_error.contour(X3_error.T, levels=[0.5, 1.0, 2.0], colors=['yellow', 'orange', 'red'],
                     extent=[-3, 3, -3, 3], linewidths=1)
    
    # Row 3: Summary
    ax_summary = fig.add_subplot(gs[2, 2:4])
    ax_summary.axis('off')
    
    stats_text = f"""
MECHANISM LEARNING SUMMARY
═══════════════════════════════════════

Single-Parent Mechanisms:
  • X1 → X2 (Linear):     MSE = {mse_x2:.4f}  {'✓' if mse_x2 < 0.5 else '✗'}
  • X4 → X5 (Quadratic):  MSE = {mse_x5:.4f}  {'✓' if mse_x5 < 0.5 else '✗'}

Multi-Parent (Collider) Mechanism:
  • X1,X2 → X3:           MSE = {X3_error.mean()**2:.4f}  {'✓' if X3_error.mean() < 0.5 else '✗'}
  
  X3 Error Statistics:
    - Mean Absolute Error: {X3_error.mean():.4f}
    - Max Absolute Error:  {X3_error.max():.4f}
    - Error Std Dev:       {X3_error.std():.4f}

Root Distributions:
  • X1: Student μ={pred_x1.mean():.2f}, σ={pred_x1.std():.2f} (Truth: μ=0, σ=1)
  • X4: Student μ={pred_x4.mean():.2f}, σ={pred_x4.std():.2f} (Truth: μ=2, σ=1)
"""
    
    ax_summary.text(0.05, 0.95, stats_text, transform=ax_summary.transAxes,
                    fontsize=10, family='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved mechanism contrast to: {output_path}")
    return output_path


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_summary(data):
    """Print run summary to console."""
    print("\n" + "="*60)
    print("RUN SUMMARY")
    print("="*60)
    
    if 'node_losses' in data:
        nl = data['node_losses']
        final = nl.iloc[-1]
        print(f"\nFinal Mechanism Losses:")
        for node in ['X1', 'X2', 'X3', 'X4', 'X5']:
            col = f'loss_{node}'
            if col in final:
                loss = final[col]
                status = '✓' if loss < 0.5 else '✗'
                print(f"  {node}: {loss:.4f} {status}")
    
    if 'metrics' in data:
        metrics = data['metrics']
        target_counts = metrics['target'].value_counts()
        print(f"\nIntervention Distribution:")
        for target, count in target_counts.items():
            pct = count / len(metrics) * 100
            print(f"  {target}: {count} ({pct:.1f}%)")
    
    print("="*60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ACE Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py run_20260113_060752           # Generate dashboard
  python visualize.py run_20260113_060752 -o out.png  # Custom output path
  python visualize.py --demo                        # Demo with untrained model
        """
    )
    parser.add_argument('run_dir', type=str, nargs='?', default=None,
                        help='Path to run directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for visualization')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode: generate mechanism contrast with untrained model')
    
    args = parser.parse_args()
    
    if args.demo:
        if not TORCH_AVAILABLE:
            print("Error: torch required for demo mode")
            return 1
        
        # Import model classes
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ace_experiments import GroundTruthSCM, StudentSCM
        
        oracle = GroundTruthSCM()
        student = StudentSCM(oracle)
        results_dir = args.run_dir or '.'
        
        print("Running in DEMO mode with untrained student...")
        create_mechanism_contrast(oracle, student, results_dir, "mechanism_contrast_demo.png")
        return 0
    
    if not args.run_dir:
        parser.print_help()
        return 1
    
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory not found: {args.run_dir}")
        return 1
    
    print(f"Loading data from: {args.run_dir}")
    data = load_run_data(args.run_dir)
    
    if not data:
        print("Error: No data files found in run directory")
        return 1
    
    print(f"Found data files: {list(data.keys())}")
    
    # Generate dashboard
    create_success_dashboard(data, args.run_dir, args.output)
    
    # Print summary
    print_summary(data)
    
    return 0


if __name__ == '__main__':
    exit(main())
