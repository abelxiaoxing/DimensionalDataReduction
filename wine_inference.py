from numpy import loadtxt, random, mean, std
from torch import manual_seed, device, cuda
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from os import path
from time import time

# 导入必要的类
from utils.neural_reduction import NeuralDimensionalityReducer

def load_wine_data(file_path):
    """加载Wine数据集"""
    data = loadtxt(file_path, delimiter=',')
    X = data[:, 1:]  # 特征
    y = data[:, 0].astype(int)  # 标签
    return X, y

def main():
    # 设置随机种子
    random.seed(42)
    manual_seed(42)
    
    # 数据路径
    data_path = path.join("datas", "wine.data")
    
    # 如果在wine子目录中运行
    if not path.exists(data_path):
        data_path = path.join("wine", "datas", "wine.data")
    
    # 模型权重路径
    weights_path = path.join("models", "wine.pth")
    
    # 加载数据
    print("加载数据...")
    X, y = load_wine_data(data_path)
    
    # 数据标准化
    X_mean = mean(X, axis=0)
    X_std = std(X, axis=0)
    X = (X - X_mean) / X_std
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    # 创建VAE降维器
    print("加载VAE模型...")
    dev = device('cuda' if cuda.is_available() else 'cpu')
    reducer = NeuralDimensionalityReducer(
        latent_dim=3,
        model_type='vae',
        device=dev,
        hidden_dims=[512, 256, 128],
        beta=1.0
    )
    
    # 初始化模型
    reducer._init_model(X_train.shape[1])
    
    # 加载预训练权重
    try:
        print(f"尝试加载预训练权重: {weights_path}")
        reducer.load_weights(weights_path)
    except Exception as e:
        print(f"加载预训练权重失败: {str(e)}")
        return
    
    # 使用VAE进行降维
    print("使用VAE进行降维...")
    start_time = time()
    X_train_reduced = reducer.transform(X_train)
    X_test_reduced = reducer.transform(X_test)
    transform_time = time() - start_time
    print(f"降维耗时: {transform_time:.4f}秒")
    
    # 训练SVM分类器
    print("训练SVM分类器...")
    start_time = time()
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(X_train_reduced, y_train)
    train_time = time() - start_time
    print(f"SVM训练耗时: {train_time:.4f}秒")
    
    # 在测试集上评估
    print("在测试集上评估...")
    y_pred = svm.predict(X_test_reduced)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
if __name__ == "__main__":
    main()