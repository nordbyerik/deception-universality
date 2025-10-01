import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('results_05.csv')

# Separate validation and test data
validation = data[data['dataset'] == 'validation'].sort_values('layer')
test = data[data['dataset'] == 'test'].sort_values('layer')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Logistic Probe Performance Across Layers', fontsize=16, fontweight='bold')

# Plot 1: Accuracy
ax1 = axes[0, 0]
width = 0.35
x = np.arange(len(validation['layer']))
ax1.bar(x - width/2, validation['accuracy'], width, label='Validation', color='#8b5cf6')
ax1.bar(x + width/2, test['accuracy'], width, label='Test', color='#3b82f6')
ax1.set_xlabel('Layer', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_title('Accuracy by Layer', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(validation['layer'])
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend()
ax1.set_ylim(0, 1.05)

# Plot 2: F1 Score
ax2 = axes[0, 1]
ax2.bar(x - width/2, validation['f1'], width, label='Validation', color='#10b981')
ax2.bar(x + width/2, test['f1'], width, label='Test', color='#06b6d4')
ax2.set_xlabel('Layer', fontsize=11)
ax2.set_ylabel('F1 Score', fontsize=11)
ax2.set_title('F1 Score by Layer', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(validation['layer'])
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()
ax2.set_ylim(0, 1.05)

# Plot 3: Confidence
ax3 = axes[1, 0]
ax3.bar(x - width/2, validation['confidence'], width, label='Validation', color='#f59e0b')
ax3.bar(x + width/2, test['confidence'], width, label='Test', color='#ef4444')
ax3.set_xlabel('Layer', fontsize=11)
ax3.set_ylabel('Confidence', fontsize=11)
ax3.set_title('Confidence by Layer', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(validation['layer'])
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend()
ax3.set_ylim(0, 1.05)

# Plot 4: All metrics for test set
ax4 = axes[1, 1]
width_multi = 0.2
ax4.bar(x - 1.5*width_multi, test['accuracy'], width_multi, label='Accuracy', color='#3b82f6')
ax4.bar(x - 0.5*width_multi, test['precision'], width_multi, label='Precision', color='#10b981')
ax4.bar(x + 0.5*width_multi, test['recall'], width_multi, label='Recall', color='#f59e0b')
ax4.bar(x + 1.5*width_multi, test['f1'], width_multi, label='F1 Score', color='#8b5cf6')
ax4.set_xlabel('Layer', fontsize=11)
ax4.set_ylabel('Score', fontsize=11)
ax4.set_title('All Metrics - Test Set', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(test['layer'])
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend()
ax4.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('probe_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Print key observations
print("\n" + "="*60)
print("KEY OBSERVATIONS:")
print("="*60)
print(f"\n1. Best performing layers:")
print(f"   Validation: Layer {validation.loc[validation['accuracy'].idxmax(), 'layer']} "
      f"(Accuracy: {validation['accuracy'].max():.3f})")
print(f"   Test: Layer {test.loc[test['accuracy'].idxmax(), 'layer']} "
      f"(Accuracy: {test['accuracy'].max():.3f})")

print(f"\n2. Layer 13 anomaly:")
print(f"   Validation accuracy: {validation[validation['layer']==13]['accuracy'].values[0]:.3f}")
print(f"   Test accuracy: {test[test['layer']==13]['accuracy'].values[0]:.3f}")
print(f"   -> Potential overfitting or data distribution issue")

print(f"\n3. Performance recovery:")
print(f"   Final layer (23) - Validation: {validation[validation['layer']==23]['accuracy'].values[0]:.3f}, "
      f"Test: {test[test['layer']==23]['accuracy'].values[0]:.3f}")

print(f"\n4. Confidence trends:")
print(f"   Validation max confidence at layer {validation.loc[validation['confidence'].idxmax(), 'layer']}: "
      f"{validation['confidence'].max():.3f}")
print(f"   Test max confidence at layer {test.loc[test['confidence'].idxmax(), 'layer']}: "
      f"{test['confidence'].max():.3f}")
print("="*60 + "\n")