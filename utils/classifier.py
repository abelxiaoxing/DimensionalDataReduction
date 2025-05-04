import time
import copy
import numpy as np
from torch import FloatTensor, LongTensor, cuda, randn_like, cat, no_grad, max
from torch import device as torch_device
from torch.nn import ModuleList, Module, Linear, BatchNorm1d, GELU, SELU, Dropout, Sequential, CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.init import kaiming_normal_, constant_
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class MLP(ModuleList):
    def __init__(self, channels, skips=None, use_bn=True, act=GELU, dropout=0.0):
        super().__init__()
        self.num_layers = len(channels) - 1
        if skips is None:
            skips = {}
        self.skips = skips
        self.channels = channels
        for i in range(1, self.num_layers + 1):
            in_channels = channels[i - 1] + (channels[skips[i]] if i in skips else 0)
            layers = [Linear(in_channels, channels[i])]
            if i < self.num_layers:
                if use_bn:
                    layers.append(BatchNorm1d(channels[i]))
                layers.append(act())
            if i + 1 == self.num_layers and dropout > 0:
                layers.append(Dropout(dropout, inplace=True))
            self.append(Sequential(*layers))

    def forward(self, x):
        xs = [x]
        for i in range(self.num_layers):
            if i + 1 in self.skips:
                x = cat([xs[self.skips[i + 1]], x], dim=-1)
            x = self[i](x)
            xs.append(x)
        return x


class ResMLP(Module):
    def __init__(self, input_size, output_size):
        super(ResMLP, self).__init__()
        self.net = MLP(
            [input_size, 16*32, 16*32, 8*32, 8*32, 4*32, 4*32, output_size],
            act=SELU,
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.net(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                kaiming_normal_(m.weight, nonlinearity='selu')
                if m.bias is not None:
                    constant_(m.bias, 0)

# PyTorch classifier wrapper
class PyTorchMLPClassifier:
    def __init__(self, input_size, num_classes, batch_size=128, max_epochs=100, patience=10, device=None):
        self.input_size = input_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device if device else torch_device('cuda' if cuda.is_available() else 'cpu')
        self.model = None
        self.criterion = CrossEntropyLoss()
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # If validation set is not provided, use a subset of training data
        if X_val is None or y_val is None:
            split_idx = int(0.8 * len(X_train))
            indices = np.random.permutation(len(X_train))
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train, y_train = X_train[train_idx], y_train[train_idx]
        
        # Convert data to PyTorch tensors
        X_train_tensor = FloatTensor(X_train).to(self.device)
        y_train_tensor = LongTensor(y_train).to(self.device)
        X_val_tensor = FloatTensor(X_val).to(self.device)
        y_val_tensor = LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = ResMLP(self.input_size, self.num_classes).to(self.device)
        
        # Initialize optimizer and criterion
        optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_dataset)
            
            self.model.eval()
            with no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                _, val_preds = max(val_outputs, 1)
                val_accuracy = (val_preds == y_val_tensor).sum().item() / len(y_val_tensor)
            
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        X_tensor = FloatTensor(X).to(self.device)
        self.model.eval()
        with no_grad():
            outputs = self.model(X_tensor)
            _, predictions = max(outputs, 1)
        return predictions.cpu().numpy()

class ClassifierEvaluator:
    def __init__(self):
        self.classifiers = {
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        }
        self.torch_mlp = None

    def evaluate(self, X_train, X_test, y_train, y_test, classifier_type=None):
        results = {}
        
        # 当指定了分类器类型时，只执行指定的分类器
        # 当未指定分类器类型时，执行所有分类器
        
        # 评估sklearn的神经网络分类器
        if classifier_type is None or classifier_type == 'NeuralNetwork':
            for name, clf in self.classifiers.items():
                start_time = time.time()
                clf.fit(X_train, y_train)
                training_time = time.time() - start_time
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'accuracy': accuracy,
                    'training_time': training_time
                }
                print(f"{name} results - Accuracy: {accuracy:.4f}, Training time: {training_time:.2f}s")
        
        # 评估自定义PyTorch MLP分类器
        if classifier_type is None or classifier_type == 'Custom MLP':
            unique_labels = np.unique(y_train)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y_train_mapped = np.array([label_map[label] for label in y_train])
            y_test_mapped = np.array([label_map[label] for label in y_test])
            
            input_size = X_train.shape[1]
            num_classes = len(unique_labels)
            
            print(f"Training custom PyTorch MLP with input size: {input_size}, num classes: {num_classes}")
            self.torch_mlp = PyTorchMLPClassifier(
                input_size=input_size,
                num_classes=num_classes,
                batch_size=128,
                max_epochs=100,
                patience=10
            )
            
            start_time = time.time()
            # Fit the model before predicting
            self.torch_mlp.fit(X_train, y_train_mapped)
            training_time = time.time() - start_time
            
            y_pred = self.torch_mlp.predict(X_test)
            y_pred_original = np.array([unique_labels[pred] for pred in y_pred])
            accuracy = accuracy_score(y_test, y_pred_original)
            
            results['Custom MLP (PyTorch)'] = {
                'accuracy': accuracy,
                'training_time': training_time
            }
            
            print(f"Custom MLP (PyTorch) results - Accuracy: {accuracy:.4f}, Training time: {training_time:.2f}s")
            
        return results