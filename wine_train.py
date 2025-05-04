import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.datasets import WineDataLoader
from utils.dimension_reduction import apply_dimensionality_reduction
from utils.visualize import Visualizer
from utils.classifier import ClassifierEvaluator


def main():
    file_path = os.path.join('datas','wine.data')
    # 设置参数
    n_components = 5
    np.random.seed(42)
    # 加载和预处理数据
    loader = WineDataLoader(file_path)
    X, y = loader.load()
    print(f"加载的数据形状: {X.shape}, 标签形状: {y.shape}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
    
    # 应用降维
    print("\n应用多种降维方法...")

    reduced_train, reduced_test, metrics, components = apply_dimensionality_reduction(
        X_train, X_test, n_components=n_components
    )
                
    # 对每种降维方法评估分类器
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
    
    
    # 可视化2D降维结果
    viz = Visualizer()
    for method_name, data in reduced_train.items():
        if data.shape[1] >= 2:
            viz.visualize_2d_data(data, y_train, f'Wine {method_name} 2D Visualization')
            print(f"Wine {method_name} 2D 可视化完成")
    # 可视化组件
    for method_name, comps in components.items():
        if method_name in ['PCA', 'NMF', 'TruncatedSVD']:
            viz.visualize_components(comps, n_components, f'{method_name} Components')
            print(f"{method_name} 组件可视化完成")
    
    # 打印性能指标
    print("\n降维性能指标:")
    for method_name, method_metrics in metrics.items():
        print(f"  {method_name}:")
        print(f"    转换时间: {method_metrics['transform_time']:.4f} 秒")
        if method_metrics['reconstruction_error'] is not None:
            print(f"    重构误差 (MSE): {method_metrics['reconstruction_error']:.4f}")
        else:
            print(f"    重构误差: 不适用")
        print(f"    降维比率: {method_metrics['reduction_ratio']:.4f}")
    
    print("\n分类器性能指标:")
    for method_name, clf_results in classifier_results.items():
        print(f"  {method_name}:")
        for clf_name, metrics in clf_results.items():
            print(f"    {clf_name}: 准确率 = {metrics['accuracy']:.4f}, 训练时间 = {metrics['training_time']:.4f} 秒")

# 执行主函数
if __name__ == "__main__":
    main()