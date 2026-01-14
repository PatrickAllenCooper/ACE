#!/usr/bin/env python3
"""
ACE Run Visualization Script
Generates comprehensive visualizations to verify experiment success/failure.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse
import os
from pathlib import Path


def load_data(run_dir):
    """Load all CSV data from a run directory."""
    data = {}
    
    node_losses_path = os.path.join(run_dir, "node_losses.csv")
    if os.path.exists(node_losses_path):
        data['node_losses'] = pd.read_csv(node_losses_path)
    
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        data['metrics'] = pd.read_csv(metrics_path)
    
    value_diversity_path = os.path.join(run_dir, "value_diversity.csv")
    if os.path.exists(value_diversity_path):
        data['value_diversity'] = pd.read_csv(value_diversity_path)
    
    dpo_path = os.path.join(run_dir, "dpo_training.csv")
    if os.path.exists(dpo_path):
        data['dpo'] = pd.read_csv(dpo_path)
    
    return data


def create_success_dashboard(data, run_dir, output_path=None):
    """Create a comprehensive dashboard showing run success metrics."""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'ACE Run Analysis: {os.path.basename(run_dir)}', fontsize=14, fontweight='bold')
    
    # Define grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # 1. Node Losses Over Time (Top Left - Large)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    if 'node_losses' in data:
        nl = data['node_losses']
        
        # Get per-episode final losses
        episode_losses = nl.groupby('episode').last().reset_index()
        
        colors = {'loss_X1': '#2ecc71', 'loss_X2': '#3498db', 'loss_X3': '#e74c3c', 
                  'loss_X4': '#9b59b6', 'loss_X5': '#f39c12'}
        labels = {'loss_X1': 'X1 (root)', 'loss_X2': 'X2 (X1→X2)', 'loss_X3': 'X3 (COLLIDER)', 
                  'loss_X4': 'X4 (root)', 'loss_X5': 'X5 (X4→X5)'}
        
        for col in ['loss_X1', 'loss_X2', 'loss_X3', 'loss_X4', 'loss_X5']:
            if col in episode_losses.columns:
                ax1.plot(episode_losses['episode'], episode_losses[col], 
                        label=labels[col], color=colors[col], linewidth=2, alpha=0.8)
        
        # Add success threshold line for X3
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='X3 Target (0.5)')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Mechanism Loss')
        ax1.set_title('Node Mechanism Losses Over Training')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # 2. Final Loss Summary (Top Right)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    
    if 'node_losses' in data:
        nl = data['node_losses']
        final_losses = nl.iloc[-1]
        
        nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
        losses = [final_losses.get(f'loss_{n}', 0) for n in nodes]
        
        # Color based on success criteria
        bar_colors = []
        for n, l in zip(nodes, losses):
            if n == 'X3':
                bar_colors.append('#27ae60' if l < 0.5 else '#e74c3c')  # Green if < 0.5
            elif n == 'X2':
                bar_colors.append('#27ae60' if l < 0.5 else '#e74c3c')  # Should be low
            elif n in ['X1', 'X4']:
                bar_colors.append('#3498db')  # Roots - neutral
            else:
                bar_colors.append('#27ae60' if l < 0.5 else '#f39c12')
        
        bars = ax2.bar(nodes, losses, color=bar_colors, edgecolor='black', linewidth=1.2)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Target')
        
        # Add value labels on bars
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            ax2.annotate(f'{loss:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('Final Loss')
        ax2.set_title('Final Mechanism Losses')
        ax2.set_yscale('log')
        
        # Add legend
        success_patch = mpatches.Patch(color='#27ae60', label='Success (< 0.5)')
        fail_patch = mpatches.Patch(color='#e74c3c', label='Failure (≥ 0.5)')
        ax2.legend(handles=[success_patch, fail_patch], loc='upper right', fontsize=8)
    
    # =========================================================================
    # 3. Intervention Distribution (Middle Left)
    # =========================================================================
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
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
    
    # =========================================================================
    # 4. X2 Value Diversity Over Episodes (Middle Center)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    if 'value_diversity' in data:
        vd = data['value_diversity']
        x2_data = vd[vd['node'] == 'X2']
        
        if len(x2_data) > 0:
            # Scatter plot with episode on x-axis
            scatter = ax4.scatter(x2_data['episode'], x2_data['value'], 
                                  alpha=0.3, s=10, c=x2_data['episode'], cmap='viridis')
            
            # Add reference line at 1.5 (the single-value trap)
            ax4.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Trap value (1.5)')
            
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('X2 Intervention Value')
            ax4.set_title('X2 Value Diversity Over Time')
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)
    
    # =========================================================================
    # 5. X2 Value Distribution Histogram (Middle Right)
    # =========================================================================
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
    
    # =========================================================================
    # 6. X3 Loss Trajectory (Bottom Left)
    # =========================================================================
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
            
            # Mark success region
            ax6.fill_between(episode_losses['episode'], 0, 0.5, alpha=0.1, color='green')
            
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('X3 Mechanism Loss')
            ax6.set_title('X3 Collider Learning Progress')
            ax6.legend(loc='upper right', fontsize=8)
            ax6.grid(True, alpha=0.3)
            
            # Annotate final value
            final_x3 = episode_losses['loss_X3'].iloc[-1]
            ax6.annotate(f'Final: {final_x3:.3f}', 
                        xy=(episode_losses['episode'].iloc[-1], final_x3),
                        xytext=(-50, 20), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black'),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # =========================================================================
    # 7. Success Criteria Summary (Bottom Center + Right)
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    # Calculate success metrics
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
        x1_pct = target_counts.get('X1', 0) / total * 100
        
        success_metrics.append(('X2 Interventions > 20%', x2_pct > 20, f'{x2_pct:.1f}%'))
        success_metrics.append(('Intervention Balance (X2 < 80%)', x2_pct < 80, f'{x2_pct:.1f}%'))
    
    if 'value_diversity' in data:
        vd = data['value_diversity']
        x2_vals = vd[vd['node'] == 'X2']['value']
        if len(x2_vals) > 0:
            std = x2_vals.std()
            success_metrics.append(('X2 Value Diversity (std > 1.0)', std > 1.0, f'σ={std:.2f}'))
    
    # Create summary table
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
    
    # Overall verdict
    all_passed = all(passed for _, passed, _ in success_metrics)
    critical_passed = success_metrics[0][1] if success_metrics else False  # X3 learned
    
    verdict_color = '#27ae60' if all_passed else ('#f39c12' if critical_passed else '#e74c3c')
    verdict_text = 'ALL CRITERIA MET' if all_passed else ('PRIMARY GOAL MET (with issues)' if critical_passed else 'FAILED')
    
    ax7.text(0.5, 0.1, verdict_text, fontsize=16, fontweight='bold', color=verdict_color,
             ha='center', va='center', transform=ax7.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=verdict_color, alpha=0.2))
    
    # Save figure
    if output_path is None:
        output_path = os.path.join(run_dir, "success_verification.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved visualization to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Visualize ACE experiment run results')
    parser.add_argument('run_dir', type=str, help='Path to run directory')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for visualization')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory not found: {args.run_dir}")
        return 1
    
    print(f"Loading data from: {args.run_dir}")
    data = load_data(args.run_dir)
    
    if not data:
        print("Error: No data files found in run directory")
        return 1
    
    print(f"Found data files: {list(data.keys())}")
    
    output_path = create_success_dashboard(data, args.run_dir, args.output)
    
    # Print summary to console
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
    
    return 0


if __name__ == '__main__':
    exit(main())
