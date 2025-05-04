from os import path, makedirs, getcwd
from numpy import random, unique, vstack, concatenate, mean, square, log10, float32 as np_float
from pandas import DataFrame
from torch import manual_seed, cuda, device
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
from utils.datasets import RamanDataLoader
from utils.visualize import Visualizer
from utils.classifier import ClassifierEvaluator
from utils.dimension_reduction import apply_dimensionality_reduction, PCAReducer
filterwarnings('ignore')

def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    random.seed(random_state)
    unique_labels, counts = unique(y, return_counts=True)
    small_class_labels = {label for label, count in zip(unique_labels, counts) if count < 3}

    # 将小样本类全部放入训练集
    X_train_small = [x for x, label in zip(X, y) if label in small_class_labels]
    y_train_small = [label for label in y if label in small_class_labels]

    # 大样本类正常划分
    mask = [label not in small_class_labels for label in y]
    X_big = X[mask]
    y_big = y[mask]

    X_train_big, X_test, y_train_big, y_test = train_test_split(
        X_big, y_big, test_size=test_size, random_state=random_state, stratify=y_big
    )

    # 合并大小样本训练集
    X_train = vstack((X_train_big, X_train_small)) if X_train_small else X_train_big
    y_train = concatenate((y_train_big, y_train_small)) if y_train_small else y_train_big

    return X_train, X_test, y_train, y_test

def main():
    data_dir = path.join("datas", "excellent_oriented")
    makedirs(path.join(getcwd(), "outputs"), exist_ok=True)
    # 设置参数
    data_type = 'Processed'  # 可选'RAW'或'Processed'
    raman_shift_range = (100, 900)
    n_samples = 1000
    n_components_list = [30]  # 目标降维维度
    random_seed = 42
    random.seed(random_seed)
    manual_seed(random_seed)
    if cuda.is_available():
        cuda.manual_seed_all(random_seed)
    
    # 检查GPU是否可用
    dev = device('cuda' if cuda.is_available() else 'cpu')
    print(f"使用设备: {dev}")
    
    # 加载和预处理数据
    print("正在加载拉曼光谱数据...")
    loader = RamanDataLoader(data_dir, data_type, raman_shift_range, n_samples)
    X, y = loader.load()

    # 替代 filtered_label_names 的方式：
    unique_labels_list = unique(y)
    label_to_name = {i: str(label) for i, label in enumerate(unique_labels_list)}
    filtered_label_names = [label_to_name[i] for i in range(len(unique_labels_list))]
    
    # 数据分割
    print("开始数据预处理...")
    preprocess_start = time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = custom_train_test_split(X_scaled, y, test_size=0.1, random_state=random_seed)
    preprocess_time = time() - preprocess_start
    print(f"数据预处理时间: {preprocess_time:.4f}秒")
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    
    # 应用多种降维方法
    print("\n应用多种降维方法...")
    all_results = []
    
    for n_components in n_components_list:
        print(f"\n降维至 {n_components} 维:")
        # 创建只包含 PCA 的 reducers 字典
        pca_reducers = {
            'PCA': PCAReducer(n_components)
        }
        
        # 记录降维训练时间
        dim_reduction_start = time()
        reduced_train, reduced_test, metrics, components = apply_dimensionality_reduction(
            X_train, X_test, n_components=n_components, reducers=pca_reducers
        )
        dim_reduction_time = time() - dim_reduction_start
        print(f"降维训练时间: {dim_reduction_time:.4f}秒")
        
        # 评估每种降维方法
        for method_name in reduced_train.keys():
            print(f"\n使用 {method_name} 降维后的数据评估...")
            
            # 使用分类器评估器
            evaluator = ClassifierEvaluator()

            # 获取当前降维方法的训练和测试数据
            X_train_reduced = reduced_train[method_name]
            X_test_reduced = reduced_test.get(method_name)
            
            # 评估NeuralNetwork分类器
            # 记录分类器训练时间
            train_start = time()
            classifier_results = evaluator.evaluate(X_train_reduced, X_test_reduced, y_train, y_test, classifier_type='NeuralNetwork')
            train_time = time() - train_start
            print(f"分类器训练总时间: {train_time:.4f}秒")
            
            # 记录推理时间
            inference_start = time()
            for clf_name, clf_result in classifier_results.items():
                if clf_name == 'NeuralNetwork':
                    evaluator.classifiers[clf_name].predict(X_test_reduced)
                elif clf_name == 'Custom MLP (PyTorch)':
                    evaluator.torch_mlp.predict(X_test_reduced)
            inference_time = time() - inference_start
            print(f"推理时间: {inference_time:.4f}秒")
            

            # 评估重构误差
            mse = None
            snr = None
            if reduced_test.get(method_name) is not None:
                print(f"评估 {method_name} 重构性能...")
                # 获取重构误差
                reconstruction_error = metrics[method_name].get('reconstruction_error')
                
                # 获取预计算的SNR (对于神经网络方法)
                precalculated_snr = metrics[method_name].get('snr')
                
                if reconstruction_error is not None:
                    # 保存MSE值
                    mse = reconstruction_error
                    
                    # 如果没有预计算的SNR，则计算信噪比(SNR)
                    if precalculated_snr is None:
                        # SNR = 10 * log10(signal_power / noise_power)
                        signal_power = mean(square(X_test))  # 使用 np.square 替代 ** 2
                        noise_power = reconstruction_error  # MSE就是噪声功率
                        snr = 10 * log10(signal_power / noise_power) if noise_power > 0 else np_float('inf')
                    else:
                        snr = precalculated_snr
                    
                    print(f"重构误差 (MSE): {reconstruction_error:.4f}")
                    print(f"信噪比 (SNR): {snr:.2f} dB")
            
            # 收集结果
            for clf_name, clf_result in classifier_results.items():
                result = {
                    'method': f"{method_name}_{clf_name}",
                    'n_components': n_components,
                    'accuracy': clf_result['accuracy'],
                    'mse': mse,
                    'snr': snr, 
                    'reduction_ratio': metrics[method_name]['reduction_ratio'],
                    'preprocess_time': preprocess_time,
                    'dim_reduction_time': dim_reduction_time,
                    'training_time': clf_result['training_time'],
                    'inference_time': inference_time
                }
                
                all_results.append(result)
    
    
    print("\n所有模型结果:")
    for result in all_results:
        method = result['method']
        n_components = result['n_components']
        
        print(f"方法: {method}, 维度: {n_components}")
        print(f"  分类准确率: {result['accuracy']:.4f}")
        print(f"  MSE: {result['mse']:.4f}")
        print(f"  SNR: {result['snr']:.2f} dB")
        print(f"  降维比: {result['reduction_ratio']:.4f}")
        print(f"  数据预处理时间: {result['preprocess_time']:.4f}秒")
        print(f"  降维时间: {result['dim_reduction_time']:.4f}秒")
        print(f"  训练时间: {result['training_time']:.4f}秒")
        print(f"  推理时间: {result['inference_time']:.4f}秒")

    # 保存结果到CSV
    results_df = DataFrame(all_results)
    results_df.to_csv(path.join("outputs", "all_results.csv"), index=False)
    print("\n结果已保存到outputs/all_results.csv")
    
    # 保存类别映射
    label_mapping = {
        'class_id': list(range(len(filtered_label_names))),
        'material_name': filtered_label_names
    }
    DataFrame(label_mapping).to_csv(path.join("outputs", "class_mapping.csv"), index=False)
    print("\n类别映射已保存到outputs/class_mapping.csv")

if __name__ == "__main__":
    main()