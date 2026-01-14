#!/usr/bin/env python3
"""
Improved Mechanism Contrast Visualization for ACE
Addresses issues with the original visualization:
1. Adds proper legends
2. Shows joint relationships for colliders
3. Clear axis labels and titles
4. Shows prediction error heatmaps for multi-parent nodes
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ace_experiments import GroundTruthSCM, StudentSCM


def create_improved_mechanism_contrast(oracle, student, results_dir, filename="mechanism_contrast_improved.png"):
    """
    Create an improved mechanism contrast visualization with:
    - Proper legends
    - Clear axis labels
    - 2D heatmaps for collider nodes
    - Error quantification
    """
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Mechanism Comparison: Ground Truth vs Learned Student', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Create grid: 3 rows, 4 columns
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35,
                          left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    x_range = torch.linspace(-4, 4, 100)
    
    # =========================================================================
    # Row 1: Root Distributions (X1 and X4)
    # =========================================================================
    
    # X1 Root Distribution
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
    
    # X4 Root Distribution
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
    
    # =========================================================================
    # Row 1: Simple Mechanisms (X1→X2, X4→X5)
    # =========================================================================
    
    # X1 → X2 (Linear: X2 = 2*X1 + 1)
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
    
    # Calculate and show MSE
    mse_x2 = np.mean((y_true_x2 - y_pred_x2)**2)
    ax_x2.text(0.95, 0.05, f'MSE: {mse_x2:.4f}', transform=ax_x2.transAxes,
               ha='right', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='yellow' if mse_x2 < 0.5 else 'red', alpha=0.5))
    
    # X4 → X5 (Quadratic: X5 = 0.2*X4²)
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
    
    # =========================================================================
    # Row 2: X3 Collider - 1D Slices (for comparison with original)
    # =========================================================================
    
    # X1 → X3 slice (with X2=0)
    ax_x3_slice1 = fig.add_subplot(gs[1, 0])
    y_true_x3_x1 = oracle.mechanisms({'X1': x_range, 'X2': torch.zeros(100)}, 'X3').detach().numpy()
    with torch.no_grad():
        parents_list = oracle.get_parents('X3')  # ['X1', 'X2'] or ['X2', 'X1']
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
    
    # X2 → X3 slice (with X1=0)
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
    
    # =========================================================================
    # Row 2: X3 Collider - 2D Heatmaps (CRITICAL for collider understanding)
    # =========================================================================
    
    # Create 2D grid for X1 and X2
    x1_grid = torch.linspace(-3, 3, 50)
    x2_grid = torch.linspace(-3, 3, 50)
    X1_mesh, X2_mesh = torch.meshgrid(x1_grid, x2_grid, indexing='ij')
    X1_flat = X1_mesh.flatten()
    X2_flat = X2_mesh.flatten()
    
    # Ground truth X3
    X3_true = oracle.mechanisms({'X1': X1_flat, 'X2': X2_flat}, 'X3').detach().numpy().reshape(50, 50)
    
    # Student X3
    with torch.no_grad():
        p_dict = {'X1': X1_flat, 'X2': X2_flat}
        p_tensor = torch.stack([p_dict[p] for p in parents_list], dim=1)
        X3_pred = student.mechanisms['X3'](p_tensor).squeeze().detach().numpy().reshape(50, 50)
    
    # Truth heatmap
    ax_heat_true = fig.add_subplot(gs[1, 2])
    im1 = ax_heat_true.imshow(X3_true.T, extent=[-3, 3, -3, 3], origin='lower', 
                               aspect='auto', cmap='viridis')
    ax_heat_true.set_xlabel('X1')
    ax_heat_true.set_ylabel('X2')
    ax_heat_true.set_title('Ground Truth X3\n(X3 = 0.5·X1 - X2 + sin(X2))', fontsize=11)
    plt.colorbar(im1, ax=ax_heat_true, label='X3')
    
    # Student heatmap
    ax_heat_pred = fig.add_subplot(gs[1, 3])
    im2 = ax_heat_pred.imshow(X3_pred.T, extent=[-3, 3, -3, 3], origin='lower',
                               aspect='auto', cmap='viridis')
    ax_heat_pred.set_xlabel('X1')
    ax_heat_pred.set_ylabel('X2')
    ax_heat_pred.set_title('Student Learned X3\n(Neural Network)', fontsize=11)
    plt.colorbar(im2, ax=ax_heat_pred, label='X3')
    
    # =========================================================================
    # Row 3: Error Analysis
    # =========================================================================
    
    # X3 Error heatmap
    ax_error = fig.add_subplot(gs[2, 0:2])
    X3_error = np.abs(X3_true - X3_pred)
    im3 = ax_error.imshow(X3_error.T, extent=[-3, 3, -3, 3], origin='lower',
                          aspect='auto', cmap='Reds')
    ax_error.set_xlabel('X1')
    ax_error.set_ylabel('X2')
    ax_error.set_title(f'X3 Absolute Error |Truth - Student|\nMean: {X3_error.mean():.4f}, Max: {X3_error.max():.4f}', 
                       fontsize=11)
    plt.colorbar(im3, ax=ax_error, label='|Error|')
    
    # Add contour lines for error levels
    ax_error.contour(X3_error.T, levels=[0.5, 1.0, 2.0], colors=['yellow', 'orange', 'red'],
                     extent=[-3, 3, -3, 3], linewidths=1)
    
    # =========================================================================
    # Row 3: Summary Statistics
    # =========================================================================
    ax_summary = fig.add_subplot(gs[2, 2:4])
    ax_summary.axis('off')
    
    # Calculate comprehensive statistics
    stats_text = """
MECHANISM LEARNING SUMMARY
═══════════════════════════════════════

Single-Parent Mechanisms:
  • X1 → X2 (Linear):     MSE = {:.4f}  {}
  • X4 → X5 (Quadratic):  MSE = {:.4f}  {}

Multi-Parent (Collider) Mechanism:
  • X1,X2 → X3:           MSE = {:.4f}  {}
  
  X3 Error Statistics:
    - Mean Absolute Error: {:.4f}
    - Max Absolute Error:  {:.4f}
    - Error Std Dev:       {:.4f}

Root Distributions:
  • X1: Student μ={:.2f}, σ={:.2f} (Truth: μ=0, σ=1)
  • X4: Student μ={:.2f}, σ={:.2f} (Truth: μ=2, σ=1)
""".format(
        mse_x2, '✓' if mse_x2 < 0.5 else '✗',
        mse_x5, '✓' if mse_x5 < 0.5 else '✗',
        X3_error.mean()**2, '✓' if X3_error.mean() < 0.5 else '✗',
        X3_error.mean(),
        X3_error.max(),
        X3_error.std(),
        pred_x1.mean(), pred_x1.std(),
        pred_x4.mean(), pred_x4.std()
    )
    
    ax_summary.text(0.05, 0.95, stats_text, transform=ax_summary.transAxes,
                    fontsize=10, family='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Save figure
    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved improved mechanism contrast to: {output_path}")
    return output_path


def load_student_checkpoint(run_dir):
    """Load a student model checkpoint if it exists."""
    # For now, we'll need to create fresh models
    # In practice, you'd want to save/load the student state dict
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate improved mechanism contrast visualization')
    parser.add_argument('run_dir', type=str, nargs='?', default=None,
                        help='Path to run directory (optional - will use current dir)')
    parser.add_argument('--demo', action='store_true', 
                        help='Run demo with untrained student to show format')
    
    args = parser.parse_args()
    
    # Create oracle (ground truth)
    oracle = GroundTruthSCM()
    
    if args.demo:
        # Demo mode: show what visualization looks like with untrained student
        student = StudentSCM(oracle)
        results_dir = args.run_dir or '.'
        print("Running in DEMO mode with untrained student...")
        create_improved_mechanism_contrast(oracle, student, results_dir, 
                                           "mechanism_contrast_demo.png")
    else:
        print("Note: To visualize a trained model, the student checkpoint needs to be loaded.")
        print("This requires saving the student state_dict during training.")
        print("")
        print("For now, run with --demo to see the visualization format.")
        print("Usage: python visualize_mechanisms.py --demo [output_dir]")


if __name__ == '__main__':
    main()
