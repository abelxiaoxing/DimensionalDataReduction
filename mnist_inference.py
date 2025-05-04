from os import path, getcwd, makedirs
from numpy import save as np_save
from torch import device, cuda, load as torch_load
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils.datasets import MNISTDataLoader
from utils.neural_reduction import NeuralDimensionalityReducer
from utils.visualize import Visualizer
from utils.classifier import ClassifierEvaluator

def get_config():
    """返回配置参数字典"""
    config = {
        'model_path': path.join('models', 'mnist.pth'),         # 模型权重路径
        'model_type': 'vae',              # 模型类型，可选['vae', 'conv_ae']
        'latent_dim': 50,                 # 潜在空间维度
        'data_path': path.join('datas', 'mnist-test-x.knd'),  # 数据路径
        'visualize': True,                # 是否可视化结果
        'classifier': 'Custom MLP'        # 指定要评估的分类器类型，可选['NeuralNetwork','Custom MLP']
    }
    return config

def load_model(model_path, model_type, latent_dim, input_dim):
    """加载预训练模型"""
    dev = device('cuda' if cuda.is_available() else 'cpu')
    print(f"使用设备: {dev}")
    
    # 创建模型实例
    reducer = NeuralDimensionalityReducer(
        latent_dim=latent_dim,
        model_type=model_type,
        device=dev,
        hidden_dims=[512, 256, 128]
    )
    
    # 初始化模型
    reducer._init_model(input_dim)
    
    # 加载权重
    try:
        print(f"尝试加载模型权重: {model_path}")
        state_dict = torch_load(model_path, map_location=dev)
        reducer.model.load_state_dict(state_dict)
        reducer.is_fitted_ = True
        print(f"成功加载模型权重")
    except Exception as e:
        print(f"加载模型权重失败: {str(e)}")
        return None
    
    return reducer

def main():
    # 获取配置参数
    config = get_config()
    
    # 加载数据
    print(f"加载数据: {config['data_path']}")
    loader = MNISTDataLoader(config['data_path'])
    X, y = loader.load()
    print(f"数据形状: {X.shape}, 标签形状: {y.shape}")
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 加载模型
    reducer = load_model(config['model_path'], config['model_type'], config['latent_dim'], X_scaled.shape[1])
    if reducer is None:
        print("模型加载失败，退出程序")
        return
    
    # 执行降维
    print("执行降维...")
    start_time = time()
    reduced_data = reducer.transform(X_scaled)
    transform_time = time() - start_time
    
    print(f"降维完成，降维后数据形状: {reduced_data.shape}")
    print(f"降维时间: {transform_time:.4f} 秒")
    
    # 计算重构误差
    reconstruction_error = reducer._calculate_reconstruction_error(X_scaled)
    print(f"重构误差 (MSE): {reconstruction_error:.4f}")
    
    # 计算信噪比
    snr = reducer._calculate_snr(X_scaled)
    print(f"信噪比 (SNR): {snr:.4f} dB")
    
    # 评估分类性能
    print("\n评估分类性能...")
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(reduced_data, y, test_size=0.1, random_state=42)
    
    # 使用ClassifierEvaluator评估分类性能
    evaluator = ClassifierEvaluator()
    clf_results = evaluator.evaluate(X_train, X_test, y_train, y_test, classifier_type=config['classifier'])
    
    # 输出分类结果
    print("\n分类性能:")
    for clf_name, metrics in clf_results.items():
        print(f"  {clf_name}: 准确率 = {metrics['accuracy']:.4f}, 分类器训练时间 = {metrics['training_time']:.4f} 秒")
    
    # 可视化结果
    if config['visualize'] and reduced_data.shape[1] >= 2:
        viz = Visualizer()
        viz.visualize_2d_data(reduced_data, y, f'MNIST {config["model_type"]} 2D Visualization (Inference)')
        print("可视化完成")
    
    # 保存降维结果
    output_dir = path.join(getcwd(), "outputs")
    makedirs(output_dir, exist_ok=True)
    output_path = path.join(output_dir, f"{config['model_type']}_reduced_data.npy")
    np_save(output_path, reduced_data)
    print(f"降维结果已保存到: {output_path}")

if __name__ == "__main__":
    main()