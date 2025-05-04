import numpy as np
import matplotlib.pyplot as plt
import os
from pandas import DataFrame, isna
from sklearn.decomposition import PCA

class Visualizer:
    """降维可视化工具类。"""
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_2d_data(self, encoded_data, labels, title):
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(encoded_data)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=10)
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        filepath = f'{self.output_dir}/{title.replace(" ", "_").lower()}.png'
        plt.savefig(filepath)
        plt.close()

    def visualize_components(self, components, n_components, title, reshape_dim=None):
        """如果提供reshape_dim则以图像形式可视化分量，否则以条形图形式显示。"""
        n_cols = 5
        n_rows = (n_components + n_cols - 1) // n_cols
        plt.figure(figsize=(12, 2 * n_rows))
        for i in range(n_components):
            plt.subplot(n_rows, n_cols, i + 1)
            comp = components[i]
            if reshape_dim:
                img = comp.reshape(reshape_dim)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
            else:
                idx = np.arange(comp.shape[0])
                plt.bar(idx, comp)
                plt.xticks(idx)
                plt.grid(True)
            plt.title(f'Component {i+1}')
        plt.tight_layout()
        filepath = f'{self.output_dir}/{title.replace(" ", "_").lower()}.png'
        plt.savefig(filepath)
        plt.close()

    def compute_reconstruction_error(self, original, reconstructed):
        mse = np.mean((original - reconstructed) ** 2)
        return mse

    def visualize_reduced_data(self, reduced_data_dict, labels, title="降维结果"):
        plot_data = {m: data for m, data in reduced_data_dict.items() if data.shape[1] >= 2}
        if not plot_data:
            print("没有可用于可视化的2D降维数据")
            return
        n_methods = len(plot_data)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
        if n_methods == 1:
            axes = [axes]
        for ax, (method_name, data) in zip(axes, plot_data.items()):
            vis_data = data[:, :2]
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(vis_data[mask, 0], vis_data[mask, 1], label=f'类别 {label}')
            ax.set_title(f'{method_name} 结果')
            ax.legend()
            ax.grid(True)
        plt.suptitle(title)
        plt.tight_layout()
        filepath = f'{self.output_dir}/{title.replace(" ", "_").lower()}.png'
        plt.savefig(filepath)
        plt.close()
        
    def plot_comparison(self, nn_results, traditional_results, output_file):
        """Plot comparison between neural network and traditional dimensionality reduction methods."""
        # Convert to DataFrames for easier manipulation
        nn_df = DataFrame(nn_results)
        trad_df = DataFrame(traditional_results)
        
        # Group metrics by dimensionality
        grouped_metrics = {}
        for n_components in nn_df['n_components'].unique():
            nn_for_dim = nn_df[nn_df['n_components'] == n_components]
            trad_for_dim = trad_df[trad_df['n_components'] == n_components]
            grouped_metrics[n_components] = {
                'neural': nn_for_dim,
                'traditional': trad_for_dim
            }
            
        # Create figure with subplots for different metrics
        metrics = ['accuracy', 'mse', 'training_time', 'reduction_ratio']
        n_dims = len(grouped_metrics)
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(n_metrics, n_dims, figsize=(5 * n_dims, 3 * n_metrics))
        if n_dims == 1:
            axes = axes.reshape(n_metrics, 1)
            
        # Set a color palette for neural vs traditional
        neural_color = '#2C7FB8'  # Blue
        trad_color = '#D95F02'    # Orange
            
        # Plot each metric for each dimensionality
        for dim_idx, (n_components, data) in enumerate(grouped_metrics.items()):
            for metric_idx, metric in enumerate(metrics):
                ax = axes[metric_idx, dim_idx]
                
                # Skip missing metrics (e.g., MSE for t-SNE)
                nn_with_metric = data['neural'][~data['neural'][metric].isna()]
                trad_with_metric = data['traditional'][~data['traditional'][metric].isna()]
                
                if len(nn_with_metric) > 0 and len(trad_with_metric) > 0:
                    # For MSE and training_time, lower is better
                    if metric in ['mse', 'training_time']:
                        nn_mean = nn_with_metric[metric].mean()
                        trad_mean = trad_with_metric[metric].mean()
                        
                        bars = ax.bar(['神经网络', '传统方法'], [nn_mean, trad_mean], color=[neural_color, trad_color])
                        ax.set_title(f'{metric.upper()} (维度: {n_components})')
                        ax.set_ylabel('值 (越小越好)')
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            value = height
                            if metric == 'training_time':
                                value_text = f'{height:.1f}s'
                            else:
                                value_text = f'{height:.4f}'
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                    value_text, ha='center', va='bottom')
                    else:
                        # For accuracy and reduction_ratio, higher is better
                        nn_mean = nn_with_metric[metric].mean()
                        trad_mean = trad_with_metric[metric].mean()
                        
                        bars = ax.bar(['神经网络', '传统方法'], [nn_mean, trad_mean], color=[neural_color, trad_color])
                        ax.set_title(f'{metric.upper()} (维度: {n_components})')
                        ax.set_ylabel('值 (越大越好)')
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.4f}', ha='center', va='bottom')
                    
                    # Add text showing best methods in each category
                    best_nn = nn_with_metric.loc[nn_with_metric[metric].idxmax() if metric in ['accuracy', 'reduction_ratio'] 
                                              else nn_with_metric[metric].idxmin()]
                    best_trad = trad_with_metric.loc[trad_with_metric[metric].idxmax() if metric in ['accuracy', 'reduction_ratio'] 
                                                  else trad_with_metric[metric].idxmin()]
                    
                    ax.text(0.05, 0.05, f'最佳神经网络: {best_nn["method"]}', transform=ax.transAxes, fontsize=9)
                    ax.text(0.05, 0.01, f'最佳传统方法: {best_trad["method"]}', transform=ax.transAxes, fontsize=9)
                else:
                    ax.text(0.5, 0.5, f'指标不适用于所有方法', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{metric.upper()} (维度: {n_components})')
                
                # Adjust y-axis to make bar height differences more visible
                if len(nn_with_metric) > 0 and len(trad_with_metric) > 0:
                    if metric in ['mse', 'training_time']:
                        min_val = min(nn_with_metric[metric].min(), trad_with_metric[metric].min())
                        max_val = max(nn_with_metric[metric].max(), trad_with_metric[metric].max())
                        # Set y-axis to start a bit below the minimum value to emphasize differences
                        buffer = (max_val - min_val) * 0.1
                        ax.set_ylim(max(0, min_val - buffer), max_val + buffer)
                    elif metric == 'accuracy':
                        min_val = min(nn_with_metric[metric].min(), trad_with_metric[metric].min())
                        # Start y-axis from a value that makes differences more visible
                        ax.set_ylim(max(0, min_val - 0.05), 1.0)
        
        plt.suptitle('神经网络 vs. 传统方法降维性能比较', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
        plt.savefig(output_file, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Also create a table summary and save as CSV
        summary = []
        for n_components in grouped_metrics:
            # For each dimensionality
            nn_data = grouped_metrics[n_components]['neural']
            trad_data = grouped_metrics[n_components]['traditional']
            
            # For each neural network method
            for _, row in nn_data.iterrows():
                summary.append({
                    'Dimensionality': n_components,
                    'Method': row['method'],
                    'Type': 'Neural Network',
                    'Accuracy': row['accuracy'],
                    'MSE': row['mse'] if not isna(row['mse']) else 'N/A',
                    'SNR': row['snr'] if not isna(row['snr']) else 'N/A',
                    'Training Time': row['training_time'],
                    'Reduction Ratio': row['reduction_ratio']
                })
            
            # For each traditional method
            for _, row in trad_data.iterrows():
                summary.append({
                    'Dimensionality': n_components,
                    'Method': row['method'],
                    'Type': 'Traditional',
                    'Accuracy': row['accuracy'],
                    'MSE': row['mse'] if not isna(row['mse']) else 'N/A',
                    'SNR': row['snr'] if not isna(row['snr']) else 'N/A',
                    'Training Time': row['training_time'],
                    'Reduction Ratio': row['reduction_ratio']
                })
        
        # Save summary to CSV
        summary_df = DataFrame(summary)
        summary_csv = output_file.replace('.png', '_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        
        return fig