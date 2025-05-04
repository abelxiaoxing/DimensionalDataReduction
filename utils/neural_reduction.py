import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import device as torch_device
from torch import tensor, randn_like, exp, sum, pow, max, softmax, var, mean, log10, float32, no_grad, long, cuda
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin


class VariationalAutoencoder(nn.Module):
    """
    变分自编码器，用于光谱数据降维
    支持监督学习，可以添加分类分支以提高特征的判别性
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256, 128], num_classes=None):
        super(VariationalAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dims[2], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[2], latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], input_dim),
        )
        
        # 添加分类分支
        self.use_classifier = num_classes is not None and num_classes > 0
        if self.use_classifier:
            # 可以根据需要调整分类器结构
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
            
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = exp(0.5 * log_var)
        eps = randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        if self.use_classifier:
            logits = self.classifier(z)
            return x_recon, mu, log_var, z, logits
        else:
            return x_recon, mu, log_var, z


class NeuralDimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    神经网络降维的Sklearn兼容封装器
    支持监督学习，可以通过分类损失引导特征学习
    """
    def __init__(self, latent_dim=20, model_type='vae', epochs=100, batch_size=32, 
                 learning_rate=0.001, device=None, hidden_dims=None, beta=0.01, 
                 classification_weight=0.0, num_classes=None): # 添加 classification_weight 和 num_classes
        self.latent_dim = latent_dim
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if torch_device is not None else ('cuda' if cuda.is_available() else 'cpu')
        self.hidden_dims = hidden_dims if hidden_dims is not None else [512, 256, 128]
        self.beta = beta
        self.classification_weight = classification_weight
        self.num_classes = num_classes
        self.model = None
        self.is_fitted_ = False
        self.reconstruction_error_ = None
        self.snr_ = None
        self.classification_accuracy_ = None
        
    def _init_model(self, input_dim):
        self.model = VariationalAutoencoder(input_dim, self.latent_dim, self.hidden_dims, num_classes=self.num_classes if self.classification_weight > 0 else None)
        self.model.to(self.device)
    
    def _create_dataloader(self, X, y=None):
        tensor_x = tensor(X, dtype=float32)
        
        if y is not None and self.classification_weight > 0:
            tensor_y = tensor(y, dtype=torch.long)
            dataset = TensorDataset(tensor_x, tensor_y)
        else:
            dataset = TensorDataset(tensor_x)
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def fit(self, X, y=None, X_test=None, y_test=None):
        start_time = time.time()
        input_dim = X.shape[1]
        self._init_model(input_dim)
        dataloader = self._create_dataloader(X, y)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        reconstruction_loss_fn = nn.MSELoss(reduction='sum') # VAE 通常用 sum
        classification_loss_fn = None
        if self.classification_weight > 0 and y is not None and self.model_type == 'vae':
            classification_loss_fn = nn.CrossEntropyLoss()
        

        use_validation = X_test is not None
        validation_data = X_test if use_validation else X
        validation_tensor = tensor(validation_data, dtype=float32).to(self.device)
        validation_labels_tensor = None
        if y is not None and self.classification_weight > 0:
            validation_labels = y_test if use_validation and y_test is not None else y
            if validation_labels is not None:
                 validation_labels_tensor = tensor(validation_labels, dtype=torch.long).to(self.device)

        self.training_losses = []
        self.validation_losses = []
        
        best_model_state = None
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = 20
        
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            total_cls_loss = 0
            
            for batch in dataloader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                
                loss = 0
                recon_loss = 0
                kl_loss = 0
                cls_loss = 0

                if self.model_type == 'vae' and self.classification_weight > 0 and len(batch) > 1 and classification_loss_fn is not None:
                    y_batch = batch[1].to(self.device)
                    recon_x, mu, log_var, z, logits = self.model(x)
                    recon_loss = reconstruction_loss_fn(recon_x, x)
                    kl_loss = -0.5 * sum(1 + log_var - pow(mu, 2) - exp(log_var))
                    cls_loss = classification_loss_fn(logits, y_batch)
                    loss = recon_loss + self.beta * kl_loss + self.classification_weight * cls_loss

                elif self.model_type == 'vae':
                    recon_x, mu, log_var, _ = self.model(x)
                    recon_loss = reconstruction_loss_fn(recon_x, x)
                    kl_loss = -0.5 * sum(1 + log_var - pow(mu, 2) - exp(log_var))
                    loss = recon_loss + self.beta * kl_loss
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                total_recon_loss += recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss
                total_kl_loss += kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss
                total_cls_loss += cls_loss.item() if torch.is_tensor(cls_loss) else cls_loss
            avg_train_loss = total_train_loss / len(dataloader.dataset)
            avg_recon_loss = total_recon_loss / len(dataloader.dataset)
            avg_kl_loss = total_kl_loss / len(dataloader.dataset)
            avg_cls_loss = total_cls_loss / len(dataloader.dataset)
            self.training_losses.append(avg_train_loss)
            
            self.model.eval()
            val_loss = 0
            val_accuracy = 0.0
            with no_grad():
                if self.model_type == 'vae':
                    if self.classification_weight > 0 and validation_labels_tensor is not None and classification_loss_fn is not None:
                        recon_x, mu, log_var, z, logits = self.model(validation_tensor)
                        val_recon_loss = reconstruction_loss_fn(recon_x, validation_tensor)
                        val_kl_loss = -0.5 * sum(1 + log_var - pow(mu, 2) - exp(log_var))
                        val_cls_loss = classification_loss_fn(logits, validation_labels_tensor)
                        val_loss = (val_recon_loss + self.beta * val_kl_loss + self.classification_weight * val_cls_loss).item() / len(validation_data)
                        _, predicted = max(logits, 1)
                        val_accuracy = (predicted == validation_labels_tensor).sum().item() / len(validation_labels_tensor)
                    else:
                        recon_x, mu, log_var, _ = self.model(validation_tensor)
                        val_recon_loss = reconstruction_loss_fn(recon_x, validation_tensor)
                        val_kl_loss = -0.5 * sum(1 + log_var - pow(mu, 2) - exp(log_var))
                        val_loss = (val_recon_loss + self.beta * val_kl_loss).item() / len(validation_data)

            
            self.validation_losses.append(val_loss)
            scheduler.step(val_loss)
            
            log_msg = f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}"
            if self.classification_weight > 0 and classification_loss_fn is not None:
                log_msg += f", Cls: {avg_cls_loss:.4f})"
            else:
                log_msg += f")"
            log_msg += f" | Val Loss: {val_loss:.4f}"
            if self.classification_weight > 0 and validation_labels_tensor is not None:
                 log_msg += f" | Val Acc: {val_accuracy:.4f}"
            print(log_msg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                epochs_no_improve = 0
                print(f"    New best model found with Val Loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
        else:
             print("Warning: No best model state found, using the model from the last epoch.")

        self.is_fitted_ = True
        self.training_time_ = time.time() - start_time
        self.reconstruction_error_ = self._calculate_reconstruction_error(X)
        self.snr_ = self._calculate_snr(X)
        
        if self.classification_weight > 0 and y is not None and self.model_type == 'vae':
            self.classification_accuracy_ = self._calculate_classification_accuracy(X, y)
            if self.classification_accuracy_ is not None:
                print(f"Final classification accuracy on training data: {self.classification_accuracy_:.4f}")
            
        return self

    def load_weights(self, weights_path):
        from torch import load
        state_dict = load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.is_fitted_ = True
        print(f"成功从 {weights_path} 加载权重")

    def transform(self, X):
        self.model.eval()
        tensor_x = tensor(X, dtype=float32).to(self.device)
        with no_grad():
            mu, _ = self.model.encode(tensor_x)
            return mu.cpu().numpy()
   
    def inverse_transform(self, Z):
        self.model.eval()
        tensor_z = tensor(Z, dtype=float32).to(self.device)
        with no_grad():
            return self.model.decode(tensor_z).cpu().numpy()
    
    def fit_transform(self, X, y=None, X_test=None, y_test=None):
        self.fit(X, y=y, X_test=X_test, y_test=y_test)
        return self.transform(X)
    
    def _calculate_reconstruction_error(self, X):
        if not self.is_fitted_: return None
        tensor_x = tensor(X, dtype=float32).to(self.device)
        self.model.eval()
        
        with no_grad():
            if self.model_type == 'vae':
                outputs = self.model(tensor_x)
                recon_x = outputs[0] if isinstance(outputs, tuple) else outputs
            else:
                return None
                
            mse = nn.MSELoss(reduction='mean')
            error = mse(recon_x, tensor_x).item()
            
        return error
        
    def _calculate_snr(self, X):
        if not self.is_fitted_:
            return None
        tensor_x = tensor(X, dtype=float32).to(self.device)
        self.model.eval()
        
        with no_grad():
            if self.model_type == 'vae':
                outputs = self.model(tensor_x)
                recon_x = outputs[0] if isinstance(outputs, tuple) else outputs
            else:
                return None
            signal_power = var(tensor_x)
            noise_power = mean((recon_x - tensor_x) ** 2)
            if noise_power <= 1e-10 or signal_power <= 0:
                return float('-inf') if signal_power <= 0 else float('inf')
            snr = 10 * log10(signal_power / noise_power)
        return snr.item()  # 返回 Python 浮点数

    def _calculate_classification_accuracy(self, X, y):
        if not self.is_fitted_ or not hasattr(self.model, 'classifier') or not self.model.use_classifier:
            return None
            
        tensor_x = tensor(X, dtype=float32).to(self.device)
        tensor_y = tensor(y, dtype=long).to(self.device)
        
        self.model.eval()
        with no_grad():
            outputs = self.model(tensor_x)
            if not isinstance(outputs, tuple) or len(outputs) < 5:
                 print("Warning: Model output format unexpected for classification accuracy calculation.")
                 return None
            logits = outputs[4]
            _, predicted = max(logits, 1)
            accuracy = (predicted == tensor_y).sum().item() / len(tensor_y)
            
        return accuracy
            
    def predict(self, X):
        tensor_x = tensor(X, dtype=float32).to(self.device)
        self.model.eval()
        with no_grad():
            outputs = self.model(tensor_x)
            logits = outputs[4]
            _, predicted = max(logits, 1)
        return predicted.cpu().numpy()
        
    def get_classification_probabilities(self, X):
        tensor_x = tensor(X, dtype=float32).to(self.device)
        self.model.eval()
        with no_grad():
            outputs = self.model(tensor_x)
            logits = outputs[4] 
            probs = softmax(logits, dim=1)
        return probs.cpu().numpy()