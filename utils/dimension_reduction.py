import time
import numpy as np
from torch import device, cuda
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from utils.neural_reduction import NeuralDimensionalityReducer


class DimensionalityReducer:
    """降维算法的基类。"""
    def __init__(self, n_components, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.transform_time = None
        self.reconstruction_error = None
        self.reduction_ratio = None
        self.snr = None

    def fit_transform(self, X_train, X_test=None):
        raise NotImplementedError


class PCAReducer(DimensionalityReducer):
    def fit_transform(self, X_train, X_test):
        start = time.time()
        model = PCA(n_components=self.n_components, random_state=self.random_state)
        train_transformed = model.fit_transform(X_train)
        test_transformed = model.transform(X_test)
        self.transform_time = time.time() - start
        X_reconstructed = model.inverse_transform(train_transformed)
        self.reconstruction_error = mean_squared_error(X_train, X_reconstructed)
        self.reduction_ratio = self.n_components / X_train.shape[1]
        return train_transformed, test_transformed, model.components_


class NeuralReducer(DimensionalityReducer):
    def __init__(self, n_components, model_type, epochs=200, batch_size=128, learning_rate=0.001, device=None, **kwargs):
        super().__init__(n_components)
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # 使用从 torch 导入的 device 和 cuda
        self.device = device or device('cuda' if cuda.is_available() else 'cpu')
        self.model_kwargs = kwargs

    def fit_transform(self, X_train, X_test=None):
        start = time.time()
        print(f"Using device: {self.device}")
        model = NeuralDimensionalityReducer(
            latent_dim=self.n_components,
            model_type=self.model_type,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
            **self.model_kwargs
        )
        train_t = model.fit_transform(X_train)
        self.transform_time = time.time() - start
        test_t = model.transform(X_test) if X_test is not None else None

        # 使用模型计算的重构误差，而不是重新计算
        self.reconstruction_error = getattr(model, 'reconstruction_error_', None)

        # 获取信噪比
        self.snr = getattr(model, 'snr_', None)

        self.reduction_ratio = self.n_components / X_train.shape[1]
        comps = None
        if self.model_type == 'conv_ae' and hasattr(model.model, 'encoder_conv'):
            w = model.model.encoder_conv[0].weight.cpu().detach().numpy()
            comps = w.reshape(w.shape[0], -1)[:self.n_components]
        return train_t, test_t, comps


class VAEReducer(NeuralReducer):
    def __init__(self, n_components):
        super().__init__(n_components, model_type='vae', epochs=100, batch_size=128, learning_rate=1e-4, hidden_dims=[512,256,128], beta=1.0, num_classes=10)


def apply_dimensionality_reduction(X_train, X_test, n_components=2, reducers=None):
    if reducers is None:
        reducers = {
            'PCA': PCAReducer(n_components),
            'VAE': VAEReducer(n_components),
        }
    else:
        pass

    reduced_train = {}
    reduced_test = {}
    metrics = {}
    components = {}

    for name, reducer in reducers.items():
        print(f"Applying {name} dimensionality reduction...")
        train_t, test_t, comp = reducer.fit_transform(X_train, X_test)
        reduced_train[name] = train_t
        if test_t is not None:
            reduced_test[name] = test_t
        metrics[name] = {
            'transform_time': reducer.transform_time,
            'reconstruction_error': reducer.reconstruction_error,
            'reduction_ratio': reducer.reduction_ratio
        }

        # 添加SNR指标（仅适用于神经网络模型）
        if hasattr(reducer, 'snr') and reducer.snr is not None:
            metrics[name]['snr'] = reducer.snr
        if comp is not None:
            components[name] = comp

    return reduced_train, reduced_test, metrics, components