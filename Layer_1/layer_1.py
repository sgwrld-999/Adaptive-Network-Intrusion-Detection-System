import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PowerTransformer, MaxAbsScaler
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from scipy import stats
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import joblib
import time
import logging
import warnings
import json

# Setup logging and suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Layer1_VAE')

# Create directories
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

class Layer1AutoencoderVAE:
    def __init__(self, input_dim, latent_dim=6, learning_rate=1e-4, layer_sizes=None):
        """Initialize the VAE model with configurable architecture"""
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes or [64, 32]  # Default larger network
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.kde = None
        self.threshold = None
        self.fallback_threshold = None  # Added for robustness
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.build_model()
        
    def sampling(self, args):
        """Reparameterization trick for VAE"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
    def build_model(self):
        """Build the VAE with customizable architecture"""
        # Encoder
        encoder_inputs = layers.Input(shape=(self.input_dim,))
        x = encoder_inputs
        
        for size in self.layer_sizes:
            x = layers.Dense(size, activation="relu", 
                             kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # VAE latent space
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var])
        
        # Instantiate encoder
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        
        # Decoder
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = latent_inputs
        
        for size in reversed(self.layer_sizes):
            x = layers.Dense(size, activation="relu", 
                             kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        decoder_outputs = layers.Dense(self.input_dim, activation="sigmoid")(x)
        
        # Instantiate decoder
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        
        # Instantiate VAE model
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = keras.Model(encoder_inputs, outputs, name="vae")
        
        # Define VAE loss with beta parameter for KL term weighting
        beta = 1.0  # Can be adjusted to control KL weight
        reconstruction_loss = keras.losses.MeanSquaredError()(encoder_inputs, outputs)
        reconstruction_loss *= self.input_dim
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + beta * kl_loss)
        
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        
        # Custom metrics
        self.vae.metrics_names.append("reconstruction_loss")
        self.vae.metrics_names.append("kl_loss")
        self.vae.metrics.append(self.reconstruction_loss_tracker)
        self.vae.metrics.append(self.kl_loss_tracker)
        
    def train(self, X_train, X_val, epochs=100, batch_size=32):
        """Train the VAE model with early stopping and LR reduction"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7
        )
        
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=f'./logs/vae_{time.strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
        
        class VAECallback(keras.callbacks.Callback):
            def __init__(self, parent):
                super(VAECallback, self).__init__()
                self.parent = parent
                
            def on_epoch_end(self, epoch, logs=None):
                # Track separate loss components
                x_val_reconstructed = self.model.predict(X_val)
                reconstruction_loss = np.mean(np.square(X_val - x_val_reconstructed))
                z_mean, z_log_var, _ = self.parent.encoder.predict(X_val)
                kl_loss = -0.5 * np.mean(np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1))
                
                # Update metrics
                self.parent.reconstruction_loss_tracker.update_state(reconstruction_loss)
                self.parent.kl_loss_tracker.update_state(kl_loss)
                
                logs['reconstruction_loss'] = reconstruction_loss
                logs['kl_loss'] = kl_loss
        
        vae_callback = VAECallback(self)
        
        logger.info("Starting VAE training...")
        history = self.vae.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping, reduce_lr, tensorboard_callback, vae_callback],
            verbose=1
        )
        
        logger.info("VAE training completed.")
        self._set_dynamic_threshold(X_train)
        return history
    
    def _set_dynamic_threshold(self, X_data):
        """Set robust thresholds using multiple methods"""
        # Compute reconstruction errors
        _, _, z = self.encoder.predict(X_data)
        reconstructed = self.decoder.predict(z)
        mse = np.mean(np.square(X_data - reconstructed), axis=1)
        
        # Set percentile-based fallback threshold (90th-99th percentile)
        self.fallback_threshold = np.percentile(mse, 97.5)
        
        # Optimize KDE bandwidth using grid search
        param_grid = {'bandwidth': np.logspace(-2, 1, 10)}
        grid_search = GridSearchCV(KernelDensity(kernel='gaussian'), param_grid, cv=5)
        grid_search.fit(mse.reshape(-1, 1))
        
        # Use the optimized bandwidth
        best_bandwidth = grid_search.best_params_['bandwidth']
        self.kde = KernelDensity(kernel='gaussian', bandwidth=best_bandwidth).fit(mse.reshape(-1, 1))
        
        log_dens = self.kde.score_samples(mse.reshape(-1, 1))
        scores = -log_dens
        
        # Robust elbow finding
        try:
            sorted_scores = np.sort(scores)
            n_samples = len(sorted_scores)
            
            if n_samples < 10:  # Not enough samples for reliable elbow detection
                self.threshold = self.fallback_threshold
            else:
                indices = np.arange(n_samples)
                
                # Use window averaging for more stable angle calculation
                window_size = max(3, int(n_samples * 0.02))
                angles = []
                
                for i in range(window_size, n_samples - window_size):
                    # Use windowed points for more stability
                    p1 = np.array([indices[i-window_size]/n_samples, sorted_scores[i-window_size]])
                    p2 = np.array([indices[i]/n_samples, sorted_scores[i]])
                    p3 = np.array([indices[i+window_size]/n_samples, sorted_scores[i+window_size]])
                    
                    # Compute vectors
                    v1 = p2 - p1
                    v2 = p3 - p2
                    
                    # Normalize vectors
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    
                    if v1_norm > 0 and v2_norm > 0:
                        v1 = v1 / v1_norm
                        v2 = v2 / v2_norm
                        
                        # Compute angle using dot product
                        dot_product = np.dot(v1, v2)
                        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                        angles.append(angle)
                    else:
                        angles.append(0)
                
                if len(angles) > 0 and max(angles) > 0.1:  # Check if we have meaningful angles
                    elbow_idx = np.argmax(angles) + window_size
                    adaptive_threshold = sorted_scores[elbow_idx]
                    
                    # Blend with percentile-based threshold for robustness
                    self.threshold = 0.7 * adaptive_threshold + 0.3 * self.fallback_threshold
                else:
                    self.threshold = self.fallback_threshold
        except Exception as e:
            logger.warning(f"Error in threshold calculation: {e}. Using fallback threshold.")
            self.threshold = self.fallback_threshold
        
        logger.info(f"Dynamic threshold: {self.threshold:.6f} (fallback: {self.fallback_threshold:.6f})")
        
        # Visualize the threshold
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, alpha=0.6, color='blue')
        plt.axvline(x=self.threshold, color='red', linestyle='--', label=f'Threshold: {self.threshold:.6f}')
        plt.axvline(x=self.fallback_threshold, color='green', linestyle=':', label=f'Fallback: {self.fallback_threshold:.6f}')
        plt.title('Anomaly Score Distribution and Thresholds')
        plt.xlabel('Anomaly Score (-log density)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('plots/anomaly_threshold.png')
        plt.close()
    
    def detect_anomalies(self, X_data):
        """Detect anomalies in the data"""
        if self.kde is None or self.threshold is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Get latent representations and reconstructions
        _, _, z = self.encoder.predict(X_data)
        reconstructed = self.decoder.predict(z)
        
        # Compute reconstruction error (MSE)
        mse = np.mean(np.square(X_data - reconstructed), axis=1)
        
        # Compute log density and anomaly scores
        log_dens = self.kde.score_samples(mse.reshape(-1, 1))
        anomaly_scores = -log_dens
        
        # Identify anomalies
        anomaly_indices = np.where(anomaly_scores > self.threshold)[0]
        anomalies = X_data[anomaly_indices]
        
        # Compute confidence
        max_score = np.max(anomaly_scores)
        min_score = np.min(anomaly_scores)
        confidence = (anomaly_scores - min_score) / (max_score - min_score) if max_score > min_score else np.zeros_like(anomaly_scores)
        
        return anomalies, anomaly_indices, anomaly_scores, confidence
    
    def get_encoded_features(self, X_data):
        """Extract features from the encoder's latent space"""
        _, _, z = self.encoder.predict(X_data)
        return z
    
    def save_model(self, base_path='models'):
        """Save the model and artifacts"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save full VAE model
        self.vae.save(f'{base_path}/layer1_model_{timestamp}.h5')
        self.encoder.save(f'{base_path}/layer1_encoder_{timestamp}.h5')
        self.decoder.save(f'{base_path}/layer1_decoder_{timestamp}.h5')
        
        # Create symlinks to latest models
        for model_type in ['model', 'encoder', 'decoder']:
            latest_link = f'{base_path}/layer1_{model_type}.h5'
            if os.path.exists(latest_link):
                os.remove(latest_link)
            os.symlink(f'layer1_{model_type}_{timestamp}.h5', latest_link)
        
        # Save threshold and metadata
        model_config = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'layer_sizes': self.layer_sizes,
            'threshold': float(self.threshold),
            'fallback_threshold': float(self.fallback_threshold),
            'timestamp': timestamp
        }
        
        with open(f'{base_path}/layer1_config_{timestamp}.json', 'w') as f:
            json.dump(model_config, f, indent=4)
        
        # Save the KDE model
        joblib.dump(self.kde, f'{base_path}/layer1_kde_{timestamp}.pkl')
        joblib.dump(self.kde, f'{base_path}/layer1_kde.pkl')
        
        return timestamp

def analyze_features(data, save_dir='plots'):
    """Analyze feature distributions and create visualizations"""
    # Create feature distribution plots
    plt.figure(figsize=(15, 10))
    
    features = data.columns
    num_features = len(features)
    rows = int(np.ceil(num_features / 3))
    
    for i, feature in enumerate(features):
        plt.subplot(rows, 3, i+1)
        sns.histplot(data[feature], kde=True)
        plt.title(f'{feature} Distribution')
        plt.tight_layout()
    
    plt.savefig(f'{save_dir}/feature_distributions.png')
    plt.close()
    
    # Feature correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = data.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_correlations.png')
    plt.close()
    
    # Calculate feature variances
    variances = data.var().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=variances.index, y=variances.values)
    plt.title('Feature Variance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_variances.png')
    plt.close()
    
    # Identify low variance features
    low_var_threshold = 0.01
    low_var_features = variances[variances < low_var_threshold].index.tolist()
    
    return {
        'variances': variances,
        'low_var_features': low_var_features,
        'correlation_matrix': corr_matrix
    }

def find_optimal_latent_dim(X_train, X_val, input_dim, min_dim=3, max_dim=15):
    """Use k-fold cross-validation to find optimal latent dimension"""
    logger.info("Finding optimal latent dimension...")
    
    # Define candidate dimensions to test
    if input_dim <= 10:
        candidate_dims = list(range(min_dim, min(max_dim, input_dim) + 1))
    else:
        # Test a range with more focus on smaller dimensions
        candidate_dims = list(range(min_dim, min(8, input_dim // 2) + 1))
        candidate_dims += [min(d, input_dim // 2) for d in [10, 12, 15]]
    
    # Remove duplicates and sort
    candidate_dims = sorted(list(set(candidate_dims)))
    
    # Combine train and validation for k-fold
    X_combined = np.vstack([X_train, X_val])
    
    results = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    for latent_dim in candidate_dims:
        fold_losses = []
        
        for train_idx, val_idx in kf.split(X_combined):
            X_fold_train, X_fold_val = X_combined[train_idx], X_combined[val_idx]
            
            # Train a smaller model for quick evaluation
            model = Layer1AutoencoderVAE(
                input_dim=input_dim, 
                latent_dim=latent_dim,
                layer_sizes=[32, 16],  # Smaller network for quick evaluation
                learning_rate=1e-3
            )
            
            # Train with fewer epochs for efficiency
            history = model.train(
                X_fold_train, X_fold_val, 
                epochs=30, 
                batch_size=64
            )
            
            # Get the best validation loss
            best_val_loss = min(history.history['val_loss'])
            fold_losses.append(best_val_loss)
        
        # Average loss across folds
        avg_loss = np.mean(fold_losses)
        logger.info(f"Latent dim {latent_dim}: avg validation loss = {avg_loss:.6f}")
        results.append((latent_dim, avg_loss))
    
    # Find dimension with lowest loss
    results.sort(key=lambda x: x[1])
    best_dim = results[0][0]
    
    # Visualize dimension search
    dims, losses = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(dims, losses, 'o-')
    plt.axvline(x=best_dim, color='red', linestyle='--')
    plt.title(f'Latent Dimension Optimization (Best: {best_dim})')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.savefig('plots/latent_dim_optimization.png')
    plt.close()
    
    logger.info(f"Optimal latent dimension: {best_dim}")
    return best_dim

def generate_evaluation_plots(model, X_normal, X_test=None, y_test=None, save_dir='plots'):
    """Generate evaluation plots for the model"""
    # Reconstruction error distribution for normal data
    _, _, z_normal = model.encoder.predict(X_normal)
    X_normal_reconstructed = model.decoder.predict(z_normal)
    normal_mse = np.mean(np.square(X_normal - X_normal_reconstructed), axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(normal_mse, kde=True, color='blue', label='Normal')
    
    # If test data with labels is available
    if X_test is not None and y_test is not None:
        _, _, z_test = model.encoder.predict(X_test)
        X_test_reconstructed = model.decoder.predict(z_test)
        test_mse = np.mean(np.square(X_test - X_test_reconstructed), axis=1)
        
        # Separate normal and anomaly in test set
        if np.sum(y_test) > 0:  # If we have anomalies
            anomaly_mse = test_mse[y_test == 1]
            sns.histplot(anomaly_mse, kde=True, color='red', alpha=0.6, label='Anomaly')
    
    plt.axvline(x=model.threshold, color='green', linestyle='--', 
                label=f'Threshold: {model.threshold:.6f}')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{save_dir}/reconstruction_error_dist.png')
    plt.close()
    
    # If test data with labels is available, generate ROC and PR curves
    if X_test is not None and y_test is not None and np.sum(y_test) > 0:
        # Get anomaly scores for test data
        _, _, anomaly_scores, _ = model.detect_anomalies(X_test)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'{save_dir}/roc_curve.png')
        plt.close()
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, anomaly_scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(f'{save_dir}/pr_curve.png')
        plt.close()
        
        # Latent space visualization (2D projection if latent_dim > 2)
        _, _, z = model.encoder.predict(X_test)
        
        if model.latent_dim >= 2:
            plt.figure(figsize=(10, 8))
            if np.sum(y_test == 0) > 0:
                plt.scatter(z[y_test == 0, 0], z[y_test == 0, 1], c='blue', alpha=0.5, label='Normal')
            if np.sum(y_test == 1) > 0:
                plt.scatter(z[y_test == 1, 0], z[y_test == 1, 1], c='red', alpha=0.5, label='Anomaly')
            plt.title('Latent Space Visualization (First 2 Dimensions)')
            plt.xlabel('Latent Dim 1')
            plt.ylabel('Latent Dim 2')
            plt.legend()
            plt.savefig(f'{save_dir}/latent_space_2d.png')
            plt.close()

def generate_report(model, history, feature_analysis, data_info, timestamp, save_dir='reports'):
    """Generate a summary report of the model training and evaluation"""
    report = []
    report.append("=" * 80)
    report.append("LAYER 1 AUTOENCODER-VAE MODEL SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model timestamp: {timestamp}")
    report.append("\n")
    
    # Data information
    report.append("DATA INFORMATION")
    report.append("-" * 80)
    report.append(f"Total samples: {data_info['total_samples']}")
    report.append(f"Training samples: {data_info['train_samples']}")
    report.append(f"Validation samples: {data_info['val_samples']}")
    report.append(f"Input features: {data_info['n_features']}")
    report.append(f"Feature names: {', '.join(data_info['feature_names'])}")
    report.append("\n")
    
    # Feature analysis
    report.append("FEATURE ANALYSIS")
    report.append("-" * 80)
    report.append("Top 5 features by variance:")
    top_var_features = feature_analysis['variances'].head(5)
    for feature, var in top_var_features.items():
        report.append(f"  - {feature}: {var:.6f}")
    
    report.append("\nLow variance features:")
    for feature in feature_analysis['low_var_features']:
        report.append(f"  - {feature}: {feature_analysis['variances'][feature]:.6f}")
    report.append("\n")
    
    # Model architecture
    report.append("MODEL ARCHITECTURE")
    report.append("-" * 80)
    report.append(f"Input dimension: {model.input_dim}")
    report.append(f"Latent dimension: {model.latent_dim}")
    report.append(f"Hidden layer sizes: {model.layer_sizes}")
    report.append(f"Learning rate: {model.learning_rate}")
    report.append("\n")
    
    # Training performance
    report.append("TRAINING PERFORMANCE")
    report.append("-" * 80)
    report.append(f"Final training loss: {history.history['loss'][-1]:.6f}")
    report.append(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    report.append(f"Final reconstruction loss: {history.history['reconstruction_loss'][-1]:.6f}")
    report.append(f"Final KL loss: {history.history['kl_loss'][-1]:.6f}")
    report.append(f"Training epochs: {len(history.history['loss'])}")
    report.append("\n")
    
    # Anomaly detection
    report.append("ANOMALY DETECTION")
    report.append("-" * 80)
    report.append(f"Dynamic threshold: {model.threshold:.6f}")
    report.append(f"Fallback threshold: {model.fallback_threshold:.6f}")
    report.append(f"Threshold method: Kernel Density Estimation with robust elbow finding")
    report.append("\n")
    
    # Saved artifacts
    report.append("SAVED ARTIFACTS")
    report.append("-" * 80)
    report.append(f"Full model: models/layer1_model_{timestamp}.h5")
    report.append(f"Encoder model: models/layer1_encoder_{timestamp}.h5")
    report.append(f"Decoder model: models/layer1_decoder_{timestamp}.h5")
    report.append(f"KDE model: models/layer1_kde_{timestamp}.pkl")
    report.append(f"Configuration: models/layer1_config_{timestamp}.json")
    report.append("\n")
    
    # Write report to file
    with open(f"{save_dir}/layer1_report_{timestamp}.txt", "w") as f:
        f.write("\n".join(report))
    
    # Create symlink to latest report
    latest_report = f"{save_dir}/layer1_report_latest.txt"
    if os.path.exists(latest_report):
        os.remove(latest_report)
    os.symlink(f"layer1_report_{timestamp}.txt", latest_report)
    
    return report

def main():
    start_time = time.time()
    np.random.seed(42)
    tf.random.set_seed(42)
    
    logger.info("Loading dataset...")
    dataset_path = "layer1_training_data.csv"
    
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.info("Using sample data instead...")
        sample_data = """tcp.dstport_category,mbtcp.trans_id,tcp.ack,mqtt.ver,tcp.connection.synack,mbtcp.len,mqtt.conflags,mqtt.conack.flags,tcp.connection.rst,http.tls_port,tcp.srcport,tcp.connection.fin,mqtt.hdrflags
0.5,0.0,1.2316824563829088e-09,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7660298471022675,0.0,0.8
0.5,0.0,1.4780189476594907e-09,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9434339426862391,0.0,0.0
1.0,0.0,3.1824211308021596e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7660298471022675,0.0,0.8
0.5,0.0,1.2316824563829088e-09,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7660298471022675,0.0,0.8"""
        df = pd.read_csv(pd.StringIO(sample_data))
        
    # Preprocess data
    logger.info("Preprocessing data...")
    
    # Drop columns with zero variance
    var = df.var()
    zero_var_cols = var[var == 0].index.tolist()
    if zero_var_cols:
        logger.info(f"Dropping {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
        df = df.drop(columns=zero_var_cols)
    
    # Handle NaN values
    if df.isna().any().any():
        logger.info("Filling NaN values with 0")
        df = df.fillna(0)
    
    # Scale features to [0, 1] range for VAE
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(df)
    
    # Save scaler for future use
    joblib.dump(scaler, 'models/layer1_scaler.pkl')
    
    # Split data for training
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    # Analyze features
    logger.info("Analyzing features...")
    feature_analysis = analyze_features(df)
    
    # Find optimal latent dimension
    input_dim = X_train.shape[1]
    best_latent_dim = find_optimal_latent_dim(X_train, X_val, input_dim)
    
    # Define model architecture based on input size
    if input_dim <= 10:
        layer_sizes = [32, 16]
    elif input_dim <= 20:
        layer_sizes = [64, 32, 16]
    else:
        layer_sizes = [128, 64, 32]
    
    # Create and train the model
    logger.info(f"Building VAE model with latent dim {best_latent_dim}...")
    model = Layer1AutoencoderVAE(
        input_dim=input_dim,
        latent_dim=best_latent_dim,
        layer_sizes=layer_sizes
    )
    
    logger.info("Training VAE model...")
    history = model.train(X_train, X_val, epochs=200, batch_size=64)
    
    # Generate evaluation plots
    logger.info("Generating evaluation plots...")
    generate_evaluation_plots(model, X_train)
    
    # Save the model
    logger.info("Saving model...")
    timestamp = model.save_model()
    
    # Prepare data info for report
    data_info = {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'n_features': input_dim,
        'feature_names': df.columns.tolist()
    }
    
    # Generate and save report
    logger.info("Generating report...")
    report = generate_report(model, history, feature_analysis, data_info, timestamp)
    
    # Extract encoded features for Layer 2
    logger.info("Extracting encoded features for Layer 2...")
    encoded_features = model.get_encoded_features(X_scaled)
    
    # Save encoded features for Layer 2
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=[f'vae_feature_{i}' for i in range(best_latent_dim)]
    )
    encoded_df.to_csv('layer1_encoded_features.csv', index=False)
    
    # Perform anomaly detection on training data (to demonstrate)
    logger.info("Running anomaly detection on training data...")
    anomalies, anomaly_indices, anomaly_scores, confidence = model.detect_anomalies(X_scaled)
    
    # Save anomaly detection results
    anomaly_results = pd.DataFrame({
        'anomaly_score': anomaly_scores,
        'confidence': confidence,
        'is_anomaly': anomaly_scores > model.threshold
    })
    anomaly_results.to_csv('layer1_anomaly_results.csv', index=False)
    
    # Print summary of anomalies found
    anomaly_count = len(anomaly_indices)
    logger.info(f"Found {anomaly_count} potential anomalies ({(anomaly_count/len(X_scaled))*100:.2f}%)")
    
    # Print execution time
    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    
    logger.info("Layer 1 VAE processing completed successfully!")

if __name__ == "__main__":
    main()