import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.datasets import MNISTDataLoader
from utils.visualize import Visualizer
from utils.classifier import ClassifierEvaluator
from utils.dimension_reduction import apply_dimensionality_reduction


def main():
    file_path = os.path.join('datas', 'mnist-test-x.knd')
    # 设置参数
    n_components = 50  # 降维后的维度
    random_seed = 42
    np.random.seed(random_seed)
    loader = MNISTDataLoader(file_path)
    X, y = loader.load()

    print(f"加载的数据形状: {X.shape}, 标签形状: {y.shape}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_seed)
    # 应用降维
    print("\n应用多种降维方法...")
    reduced_train, reduced_test, metrics, components = apply_dimensionality_reduction(
        X_train, X_test, n_components=n_components
    )

    evaluator = ClassifierEvaluator()
    classifier_results = {}
    for method_name in reduced_train.keys():
        if method_name in reduced_test:
            print(f"使用 {method_name} 降维后的数据评估分类器...")
            clf_results = evaluator.evaluate(
                reduced_train[method_name], 
                reduced_test[method_name], 
                y_train, y_test
            )
            classifier_results[method_name] = clf_results
    
    viz = Visualizer()
    # 可视化2D降维结果
    for method_name, data in reduced_train.items():
        if data.shape[1] >= 2:  # 确保有足够的维度进行可视化
            viz.visualize_2d_data(data, y_train, f'MNIST {method_name} 2D Visualization')
            print(f"MNIST {method_name} 2D 可视化完成")
    # 可视化组件（仅对有组件的方法）
    for method_name, comps in components.items():
        if method_name in ['PCA', 'NMF', 'TruncatedSVD']:
            viz.visualize_components(comps, n_components, f'{method_name} Components', reshape_dim=(28, 28))
            print(f"{method_name} 组件可视化完成")
    
    # 输出性能比较
    print("\n降维性能指标:")
    for method_name, method_metrics in metrics.items():
        print(f"\n{method_name}:")
        print(f"  转换时间: {method_metrics['transform_time']:.4f} 秒")
        if method_metrics['reconstruction_error'] is not None:
            print(f"  重构误差 (MSE): {method_metrics['reconstruction_error']:.4f}")
        else:
            print(f"  重构误差: 不适用")
        print(f"  降维比率: {method_metrics['reduction_ratio']:.4f}")
    
    print("\n分类性能:")
    for method_name, clf_results in classifier_results.items():
        print(f"  {method_name}:")
        for clf_name, metrics in clf_results.items():
            print(f"    {clf_name}: 准确率 = {metrics['accuracy']:.4f}, 训练时间 = {metrics['training_time']:.4f} 秒")
    
if __name__ == "__main__":
    main()