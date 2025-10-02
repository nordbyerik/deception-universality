import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_probe_performance(csv_file, output_prefix='probe', return_summary=False):
    """
    Create 4 separate line plots for logistic probe performance across layers.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the results
    output_prefix : str
        Prefix for output PNG files (default: 'probe')
    return_summary : bool
        If True, return summary statistics as a dictionary
    """
    
    # Load the data
    data = pd.read_csv(csv_file)
    
    # Separate validation and test data
    validation = data[data['dataset'] == 'validation'].sort_values('layer')
    test = data[data['dataset'] == 'test'].sort_values('layer')
    
    # More muted, distinct color palette
    val_color = '#4A5568'  # Dark gray
    test_color = '#A0AEC0'  # Light gray
    
    # Plot 1: Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(validation['layer'], validation['accuracy'], 
             marker='o', linewidth=2, markersize=6, label='Validation', 
             color=val_color)
    plt.plot(test['layer'], test['accuracy'], 
             marker='s', linewidth=2, markersize=6, label='Test', 
             color=test_color)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy by Layer', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(validation['layer'], validation['f1'], 
             marker='o', linewidth=2, markersize=6, label='Validation', 
             color=val_color)
    plt.plot(test['layer'], test['f1'], 
             marker='s', linewidth=2, markersize=6, label='Test', 
             color=test_color)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score by Layer', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Confidence
    plt.figure(figsize=(10, 6))
    plt.plot(validation['layer'], validation['confidence'], 
             marker='o', linewidth=2, markersize=6, label='Validation', 
             color=val_color)
    plt.plot(test['layer'], test['confidence'], 
             marker='s', linewidth=2, markersize=6, label='Test', 
             color=test_color)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Confidence', fontsize=12)
    plt.title('Confidence by Layer', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: All metrics for test set
    plt.figure(figsize=(12, 6))
    metric_colors = ['#2C5282', '#2F855A', '#C05621', '#6B46C1']  # Muted blue, green, orange, purple
    markers = ['o', 's', '^', 'D']
    
    plt.plot(test['layer'], test['accuracy'], 
             marker=markers[0], linewidth=2, markersize=6, label='Accuracy', 
             color=metric_colors[0])
    plt.plot(test['layer'], test['precision'], 
             marker=markers[1], linewidth=2, markersize=6, label='Precision', 
             color=metric_colors[1])
    plt.plot(test['layer'], test['recall'], 
             marker=markers[2], linewidth=2, markersize=6, label='Recall', 
             color=metric_colors[2])
    plt.plot(test['layer'], test['f1'], 
             marker=markers[3], linewidth=2, markersize=6, label='F1 Score', 
             color=metric_colors[3])
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('All Metrics - Test Set', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate summary statistics
    best_val_layer = validation.loc[validation['accuracy'].idxmax(), 'layer']
    best_val_acc = validation['accuracy'].max()
    best_test_layer = test.loc[test['accuracy'].idxmax(), 'layer']
    best_test_acc = test['accuracy'].max()
    
    final_layer = validation['layer'].max()
    final_val_acc = validation[validation['layer']==final_layer]['accuracy'].values[0]
    final_test_acc = test[test['layer']==final_layer]['accuracy'].values[0]
    
    max_val_conf_layer = validation.loc[validation['confidence'].idxmax(), 'layer']
    max_val_conf = validation['confidence'].max()
    max_test_conf_layer = test.loc[test['confidence'].idxmax(), 'layer']
    max_test_conf = test['confidence'].max()
    
    summary = {
        'model': output_prefix,
        'num_layers': len(validation),
        'best_val_layer': int(best_val_layer),
        'best_val_acc': float(best_val_acc),
        'best_test_layer': int(best_test_layer),
        'best_test_acc': float(best_test_acc),
        'final_layer': int(final_layer),
        'final_val_acc': float(final_val_acc),
        'final_test_acc': float(final_test_acc),
        'max_val_conf_layer': int(max_val_conf_layer),
        'max_val_conf': float(max_val_conf),
        'max_test_conf_layer': int(max_test_conf_layer),
        'max_test_conf': float(max_test_conf),
        'avg_test_acc': float(test['accuracy'].mean()),
        'std_test_acc': float(test['accuracy'].std())
    }
    
    # Print key observations
    print("\n" + "="*60)
    print("KEY OBSERVATIONS:")
    print("="*60)
    print(f"\n1. Best performing layers:")
    print(f"   Validation: Layer {best_val_layer} (Accuracy: {best_val_acc:.3f})")
    print(f"   Test: Layer {best_test_layer} (Accuracy: {best_test_acc:.3f})")
    
    # Check if layer 13 exists before reporting on it
    if 13 in validation['layer'].values:
        layer13_val = validation[validation['layer']==13]['accuracy'].values[0]
        layer13_test = test[test['layer']==13]['accuracy'].values[0]
        print(f"\n2. Layer 13 performance:")
        print(f"   Validation accuracy: {layer13_val:.3f}")
        print(f"   Test accuracy: {layer13_test:.3f}")
        summary['layer13_val_acc'] = float(layer13_val)
        summary['layer13_test_acc'] = float(layer13_test)
    
    print(f"\n3. Final layer ({final_layer}) performance:")
    print(f"   Validation: {final_val_acc:.3f}, Test: {final_test_acc:.3f}")
    
    print(f"\n4. Confidence trends:")
    print(f"   Validation max confidence at layer {max_val_conf_layer}: {max_val_conf:.3f}")
    print(f"   Test max confidence at layer {max_test_conf_layer}: {max_test_conf:.3f}")
    
    print(f"\n5. Overall statistics:")
    print(f"   Number of layers analyzed: {len(validation)}")
    print(f"   Average test accuracy: {test['accuracy'].mean():.3f}")
    print(f"   Std dev test accuracy: {test['accuracy'].std():.3f}")
    print("="*60 + "\n")
    
    if return_summary:
        return summary


def create_summary_table(csv_files, output_prefixes, output_file='model_summary.csv'):
    """
    Create a summary table comparing multiple models.
    
    Parameters:
    -----------
    csv_files : list of str
        List of paths to CSV files
    output_prefixes : list of str
        List of prefixes for each model
    output_file : str
        Output CSV filename (default: 'model_summary.csv')
    """
    summaries = []
    
    for csv_file, prefix in zip(csv_files, output_prefixes):
        print(f"\nProcessing {prefix}...")
        summary = plot_probe_performance(csv_file, output_prefix=prefix, return_summary=True)
        summaries.append(summary)
    
    # Create DataFrame and save
    df = pd.DataFrame(summaries)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Summary table saved to {output_file}")
    print("\nSummary Table:")
    print(df.to_string(index=False))
    
    return df


# Example usage:
if __name__ == "__main__":
    # Process multiple models and create summary table
    csv_files = [
        'results/results_05.csv',
        'results/results_3.csv',
        'results/results_7.csv'
    ]
    
    output_prefixes = ['model_05', 'model_3', 'model_7']
    
    summary_df = create_summary_table(csv_files, output_prefixes, output_file='model_comparison.csv')