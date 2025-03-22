import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MaxAbsScaler
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import os
import json
from scipy import stats
from sklearn.neighbors import KernelDensity

tf.get_logger().setLevel('ERROR')

# Define residual block for Autoencoder
def residual_block(x, units):
    """Creates a residual block with two dense layers."""
    shortcut = x
    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# Define focal loss for multi-class classification
def focal_loss(gamma=2.0, alpha=None):
    """Focal loss function to handle class imbalance."""
    alpha = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.gather(y_pred, y_true, batch_dims=1)
        ce = -tf.math.log(p_t)
        alpha_t = tf.gather(alpha, y_true) if alpha is not None else 1.0
        focal_weight = alpha_t * tf.math.pow((1 - p_t), gamma)
        return focal_weight * ce
    return loss

# Data augmentation for sequences
def augment_sequence(sequence, noise_level=0.01):
    """Applies Gaussian noise to a sequence for augmentation."""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

# Sequence generator for efficient data handling
class SequenceGenerator(tf.keras.utils.Sequence):
    """Custom sequence generator with optional data augmentation."""
    def __init__(self, X, y, batch_size, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.augment:
            batch_X = np.array([augment_sequence(seq) for seq in batch_X])
        return batch_X, batch_y

class EnhancedAdaptiveNIDS:
    def __init__(self, input_dim, latent_dim=64, learning_rate=1e-4):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.model = self._build_autoencoder()
        self.kde = None
        self.threshold = None

    def _build_autoencoder(self):
        """Builds an Autoencoder with residual connections."""
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = residual_block(x, 128)  # Residual connection
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = residual_block(x, 64)  # Residual connection
        encoded = layers.Dense(self.latent_dim, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = residual_block(x, 64)  # Residual connection
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = residual_block(x, 128)  # Residual connection
        decoded = layers.Dense(self.input_dim, activation='linear')(x)
        autoencoder = keras.Model(inputs=inputs, outputs=decoded)
        autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='huber')
        return autoencoder

    def train(self, X_train, X_val, epochs=50, batch_size=64):
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        history = self.model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val, X_val), callbacks=[early_stopping, reduce_lr], verbose=1)
        self._set_dynamic_threshold(X_train)
        return history

    def _set_dynamic_threshold(self, X_data):
        """Sets a dynamic threshold using KDE (kept at 95th percentile for now)."""
        reconstructed = self.model.predict(X_data)
        errors = np.mean(np.abs(X_data - reconstructed), axis=1)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(errors.reshape(-1, 1))
        log_dens = self.kde.score_samples(errors.reshape(-1, 1))
        self.threshold = np.percentile(-log_dens, 95)  # TODO: Make adaptive based on data distribution
        print(f"Dynamic threshold set to: {self.threshold}")

    def detect_anomalies(self, X_data):
        if self.kde is None or self.threshold is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        reconstructed = self.model.predict(X_data)
        errors = np.mean(np.abs(X_data - reconstructed), axis=1)
        log_dens = self.kde.score_samples(errors.reshape(-1, 1))
        anomaly_indices = np.where(-log_dens > self.threshold)[0]
        confidence = -log_dens / np.max(-log_dens) if len(log_dens) > 0 else np.array([])
        return X_data[anomaly_indices], anomaly_indices, errors, confidence

    def get_encoded_features(self, X_data):
        encoder = keras.Model(inputs=self.model.input, outputs=self.model.layers[8].output)  # Adjust index if structure changes
        return encoder.predict(X_data)

    def get_feature_importance(self, X_data, anomaly_indices):
        """Computes feature importance based on reconstruction errors of anomalies."""
        reconstructed = self.model.predict(X_data)
        errors = np.abs(X_data - reconstructed)
        anomaly_errors = errors[anomaly_indices]
        feature_importance = np.mean(anomaly_errors, axis=0)
        return feature_importance

class AdaptiveNIDSLayer2:
    def __init__(self, input_dim, num_classes, seq_length=10):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.temperature = tf.Variable(1.0, trainable=True)
        self.model = self._build_model()
        self.class_weights = None

    def _build_model(self):
        """Builds a CNN-BiLSTM model with attention mechanism."""
        inputs = layers.Input(shape=(self.seq_length, self.input_dim))
        x = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        shortcut = layers.Conv1D(128, 1, kernel_regularizer=regularizers.l2(1e-4))(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
        shortcut = layers.MaxPooling1D(4)(shortcut)
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4)))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4)))(x)
        # Attention mechanism
        attention_scores = layers.Dense(1, activation=None)(x)
        attention_weights = layers.Softmax(axis=1)(attention_scores)
        context_vector = layers.Lambda(lambda inputs: tf.reduce_sum(inputs[0] * inputs[1], axis=1))([x, attention_weights])
        dense1 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(context_vector)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(0.25)(dense1)
        logits = layers.Dense(self.num_classes)(dense1)
        scaled_logits = layers.Lambda(lambda x: x / self.temperature)(logits)
        outputs = layers.Activation('softmax')(scaled_logits)
        return keras.Model(inputs=inputs, outputs=outputs)  # Compile in train method

    def compute_class_weights(self, y_train):
        y_train_int = y_train.astype(int)
        unique_classes = np.unique(y_train_int)
        class_counts = np.bincount(y_train_int)
        total = len(y_train_int)
        self.class_weights = {i: total / (len(unique_classes) * count) for i, count in enumerate(class_counts)}
        print("Class weights:", self.class_weights)
        return self.class_weights

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=64):
        if self.class_weights is None:
            self.compute_class_weights(y_train)
        alpha = np.array([self.class_weights.get(i, 1.0) for i in range(self.num_classes)])
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipvalue=1.0),  # Gradient clipping
            loss=focal_loss(gamma=2.0, alpha=alpha),  # Focal loss
            metrics=['accuracy']
        )
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint('best_layer2_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
            keras.callbacks.TensorBoard(log_dir=f'./logs/layer2_{time.strftime("%Y%m%d-%H%M%S")}', histogram_freq=1)
        ]
        train_generator = SequenceGenerator(X_train, y_train, batch_size, augment=True)
        val_generator = SequenceGenerator(X_val, y_val, batch_size, augment=False) if X_val is not None else None
        if val_generator:
            history = self.model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=callbacks)
        else:
            history = self.model.fit(train_generator, epochs=epochs, validation_split=0.2, callbacks=callbacks)
        return history

    def evaluate(self, X_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        report = classification_report(y_test, y_pred, output_dict=True)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')
        plt.close()
        return report, y_pred, y_pred_probs

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

def preprocess_data(file_path, test_size=0.2, random_state=42, use_adasyn=True, top_k_features=50):
    df = pd.read_csv(file_path)
    if df.empty or 'Attack_label' not in df.columns:
        raise ValueError("Dataset is empty or missing 'Attack_label' column.")
    X = df.drop(['Attack_label'], axis=1)
    y = df['Attack_label'].astype(int)
    print("Class distribution before preprocessing:", y.value_counts(normalize=True) * 100)
    # Ensemble feature selection
    if np.any(X < 0):
        score_funcs = [mutual_info_classif, f_classif]
    else:
        score_funcs = [mutual_info_classif, chi2, f_classif]
    ranks = []
    for score_func in score_funcs:
        scores = score_func(X, y) if score_func != chi2 else score_func(X, y)[0]
        scores = np.nan_to_num(scores)
        rank = np.argsort(np.argsort(-scores))
        ranks.append(rank)
    average_rank = np.mean(ranks, axis=0)
    selected_indices = np.argsort(average_rank)[:min(top_k_features, X.shape[1])]
    selected_features = X.columns[selected_indices].tolist()
    X_selected = X.iloc[:, selected_indices]
    print(f"Selected {len(selected_features)} features: {selected_features[:5]}...")
    # Scaling
    scaler = PowerTransformer(method='yeo-johnson') if np.any(np.abs(stats.skew(X_selected)) > 1) else MaxAbsScaler()
    scaler.fit(X_selected)
    X_scaled = scaler.transform(X_selected)
    joblib.dump(scaler, 'scaler.pkl')
    # Outlier removal with adaptive contamination
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', n_jobs=min(mp.cpu_count(), 4))
    outlier_labels = lof.fit_predict(X_scaled)
    outlier_mask = outlier_labels == 1
    X_scaled = X_scaled[outlier_mask]
    y = y[outlier_mask]
    print(f"Removed {np.sum(~outlier_mask)} outliers.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    if use_adasyn and len(np.unique(y_train)) > 1:
        adasyn = ADASYN(random_state=random_state)
        X_train, y_train = adasyn.fit_resample(X_train, y_train)
        print("Class distribution after ADASYN:", pd.Series(y_train).value_counts(normalize=True) * 100)
    return X_train, X_test, y_train, y_test, scaler, selected_features

def create_sequences(data, labels, seq_length=10, stride=2, min_seq_variance=0.01):
    if len(data) < seq_length:
        raise ValueError(f"Data length ({len(data)}) < sequence length ({seq_length}).")
    effective_seq_length = seq_length if np.mean(np.var(data, axis=0)) >= min_seq_variance else max(5, seq_length // 2)
    sequences, seq_labels = [], []
    for i in range(0, len(data) - effective_seq_length + 1, stride):
        seq = data[i:i + effective_seq_length]
        sequences.append(seq)
        seq_labels.append(labels[i + effective_seq_length - 1])
    return np.array(sequences), np.array(seq_labels)

if __name__ == "__main__":
    start_time = time.time()
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    np.random.seed(42)
    tf.random.set_seed(42)
    dataset_path = "training_dataset.csv"  # Update path as needed
    X_train, X_test, y_train, y_test, scaler, selected_features = preprocess_data(dataset_path)
    
    print("\nTraining Layer 1: Autoencoder...")
    layer1 = EnhancedAdaptiveNIDS(input_dim=X_train.shape[1])
    layer1.train(X_train, X_test)
    anomalies, anomaly_indices, errors, confidence = layer1.detect_anomalies(X_test)
    print(f"Detected {len(anomaly_indices)} anomalies out of {len(X_test)} samples.")
    feature_importance = layer1.get_feature_importance(X_test, anomaly_indices)
    print("Top 5 important features for anomalies:")
    sorted_indices = np.argsort(-feature_importance)
    for i in sorted_indices[:5]:
        print(f"{selected_features[i]}: {feature_importance[i]:.4f}")
    encoded_features = layer1.get_encoded_features(anomalies)
    y_anomalies = y_test[anomaly_indices] if not isinstance(y_test, pd.Series) else y_test.iloc[anomaly_indices]
    X_layer2, y_layer2 = create_sequences(encoded_features, y_anomalies)

    X_train_l2, X_test_l2, y_train_l2, y_test_l2 = train_test_split(
        X_layer2, y_layer2, test_size=0.2, random_state=42, stratify=y_layer2 if len(np.unique(y_layer2)) > 1 else None
    )
    
    print("\nTraining Layer 2: CNN-BiLSTM...")
    layer2 = AdaptiveNIDSLayer2(input_dim=X_train_l2.shape[2], num_classes=len(np.unique(y_train_l2)))
    layer2.train(X_train_l2, y_train_l2, X_test_l2, y_test_l2)
    report, y_pred, y_pred_probs = layer2.evaluate(X_test_l2, y_test_l2)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    layer1.model.save(f'models/autoencoder_model_{timestamp}.h5')
    layer2.model.save(f'models/cnn_bilstm_model_{timestamp}.h5')
    config = {
        'dataset_path': dataset_path,
        'layer1': {'latent_dim': layer1.latent_dim, 'threshold': layer1.threshold},
        'layer2': {'seq_length': layer2.seq_length, 'class_weights': layer2.class_weights},
        'timestamp': timestamp
    }
    with open(f'models/config_{timestamp}.json', 'w') as f:
        json.dump(config, f, indent=4)
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")