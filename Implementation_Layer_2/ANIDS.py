import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

tf.keras.utils.get_custom_objects().update({'mish': tf.keras.layers.Activation(mish)})

def preprocess_data(file_path, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    y = df['Attack_label']
    X = df.drop(['Attack_label'], axis=1)
    Q1, Q3 = X.quantile(0.25), X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - 3 * IQR)) | (X > (Q3 + 3 * IQR))).any(axis=1)
    X, y = X[mask], y[mask]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'robust_scaler.pkl')
    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)

class AdaptiveNIDSLayer1:
    def __init__(self, input_dim, latent_dim=16):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = self._build_autoencoder()

    def _build_autoencoder(self):
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.BatchNormalization()(inputs)
        x = layers.Reshape((-1, 1))(x)
        conv1 = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
        conv2 = layers.Conv1D(32, 3, activation='relu', padding='same')(conv1)
        res = layers.Conv1D(32, 1, padding='same')(conv1)
        x = layers.Add()([conv2, res])
        x = layers.GlobalAveragePooling1D(name="gap_layer")(x)
        x = layers.Dense(64, activation=mish, kernel_regularizer=regularizers.l1(0.0005))(x)
        x = layers.Dropout(0.3)(x)
        encoded = layers.Dense(self.latent_dim, activation='linear')(x)
        x = layers.RepeatVector(self.input_dim)(encoded)
        x = layers.LSTM(self.latent_dim * 2, return_sequences=True, recurrent_dropout=0.25)(x)
        decoded = layers.TimeDistributed(layers.Dense(1, activation='linear'))(x)
        decoded = layers.Flatten()(decoded)
        autoencoder = keras.Model(inputs=inputs, outputs=decoded)
        autoencoder.compile(optimizer=keras.optimizers.Adam(1e-4), loss='mse')
        return autoencoder

    def train(self, X_train, X_val, epochs=50):
        lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        return self.model.fit(X_train, X_train, epochs=epochs, batch_size=64, validation_data=(X_val, X_val), callbacks=[lr_schedule, early_stopping])

    def detect_anomalies(self, X_data, threshold=0.02):
        reconstructed = self.model.predict(X_data)
        errors = np.mean(np.square(X_data - reconstructed), axis=1)
        return X_data[errors > threshold], np.where(errors > threshold)[0]

    def extract_features(self, X_anomalies):
        feature_extractor = keras.Model(inputs=self.model.input, outputs=self.model.get_layer("gap_layer").output)
        return feature_extractor.predict(X_anomalies)

def apply_pca(X_features, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold)
    X_features_pca = pca.fit_transform(X_features)
    joblib.dump(pca, 'pca_model.pkl')
    return X_features_pca, pca

def select_features(X_features, y_labels, n_features=10):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_features, y_labels)
    feature_indices = np.argsort(rf.feature_importances_)[::-1][:n_features]
    return X_features[:, feature_indices], feature_indices

def create_sequences(data, labels=None, seq_length=10, normalize=True):
    data = np.array(data)
    if normalize:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        joblib.dump(scaler, 'sequence_scaler.pkl')
    sequences, seq_labels = [], []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
        if labels is not None:
            seq_labels.append(labels[i + seq_length - 1])
    sequences = np.array(sequences)
    if labels is not None:
        return sequences, np.array(seq_labels)
    else:
        return sequences

def balance_sequences_with_smote(X_sequences, y_sequences):
    y_sequences = y_sequences.astype(int)
    original_shape = X_sequences.shape
    X_seq_2d = X_sequences.reshape(X_sequences.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_seq_balanced, y_seq_balanced = smote.fit_resample(X_seq_2d, y_sequences)
    return X_seq_balanced.reshape(-1, original_shape[1], original_shape[2]), y_seq_balanced

class AdaptiveNIDSLayer2:
    def __init__(self, input_dim, num_classes, seq_length=10):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=(self.seq_length, self.input_dim))
        x = layers.Conv1D(32, kernel_size=3, padding='same')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SpatialDropout1D(0.2)(x)
        conv2 = layers.Conv1D(64, kernel_size=3, padding='same')(x)
        conv2 = layers.LayerNormalization()(conv2)
        conv2 = layers.Activation('relu')(conv2)
        res = layers.Conv1D(64, kernel_size=1, padding='same')(x)
        x = layers.Add()([conv2, res])
        x = layers.SpatialDropout1D(0.2)(x)
        lstm_units = 32
        x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.2))(x)
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Softmax()(attention)
        attention = layers.RepeatVector(lstm_units * 2)(attention)
        attention = layers.Permute([2, 1])(attention)
        x = layers.Multiply()([x, attention])
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(48, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        initial_lr = 1e-3
        warmup_epochs = 5
        def lr_schedule(epoch):
            if epoch < warmup_epochs:
                return initial_lr * ((epoch + 1) / warmup_epochs)
            else:
                decay_epochs = epochs - warmup_epochs
                return initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / decay_epochs))
        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
        class F1ScoreCallback(keras.callbacks.Callback):
            def __init__(self, validation_data, patience=5):
                super(F1ScoreCallback, self).__init__()
                self.X_val, self.y_val = validation_data
                self.patience = patience
                self.best_f1 = 0
                self.wait = 0
                self.best_weights = None
            def on_epoch_end(self, epoch, logs={}):
                y_pred = np.argmax(self.model.predict(self.X_val), axis=1)
                f1 = f1_score(self.y_val, y_pred, average='weighted')
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.model.stop_training = True
                        self.model.set_weights(self.best_weights)
        f1_callback = F1ScoreCallback(validation_data=(X_val, y_val))
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[lr_scheduler, f1_callback])

    def evaluate_model(self, X_test, y_test):
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        detection_rate = {}
        false_positive_rate = {}
        for class_idx in range(self.num_classes):
            true_positives = cm[class_idx, class_idx]
            false_negatives = np.sum(cm[class_idx, :]) - true_positives
            false_positives = np.sum(cm[:, class_idx]) - true_positives
            true_negatives = np.sum(cm) - true_positives - false_negatives - false_positives
            detection_rate[class_idx] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            false_positive_rate[class_idx] = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        return {'classification_report': report, 'confusion_matrix': cm, 'detection_rate': detection_rate, 'false_positive_rate': false_positive_rate}

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        if y_pred.shape[-1] > 1:
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            logits = tf.math.log(y_pred)
        else:
            logits = y_pred
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        pt = tf.exp(-ce)
        loss = alpha * tf.pow(1-pt, gamma) * ce
        return tf.reduce_mean(loss)
    return focal_loss_fixed

def main():
    file_path = "/Users/siddhantgond/Desktop/6THSEM/Project_Elective/Adaptive-Network-Intrusion-Detection-System/Implementaiton/training_dataset.csv"
    X_train, X_val, y_train, y_val = preprocess_data(file_path)
    anomaly_detector = AdaptiveNIDSLayer1(input_dim=X_train.shape[1], latent_dim=16)
    history = anomaly_detector.train(X_train, X_val, epochs=50)
    X_anomalies, indices = anomaly_detector.detect_anomalies(X_train, threshold=0.02)
    y_anomalies = y_train.iloc[indices].values.astype(int)
    X_features = anomaly_detector.extract_features(X_anomalies)
    X_pca, pca = apply_pca(X_features)
    X_selected, feature_indices = select_features(X_pca, y_anomalies, n_features=10)
    joblib.dump(feature_indices, 'feature_indices.pkl')
    seq_length = 10
    X_sequences, y_sequences = create_sequences(X_selected, y_anomalies, seq_length=seq_length)
    X_balanced, y_balanced = balance_sequences_with_smote(X_sequences, y_sequences)
    X_val_anomalies, val_indices = anomaly_detector.detect_anomalies(X_val, threshold=0.02)
    y_val_anomalies = y_val.iloc[val_indices].values.astype(int)
    X_val_features = anomaly_detector.extract_features(X_val_anomalies)
    X_val_pca = pca.transform(X_val_features)
    X_val_selected = X_val_pca[:, feature_indices]
    X_val_sequences, y_val_sequences = create_sequences(X_val_selected, y_val_anomalies, seq_length=seq_length)
    num_classes = len(np.unique(y_balanced))
    layer2_model = AdaptiveNIDSLayer2(input_dim=X_balanced.shape[2], num_classes=num_classes, seq_length=seq_length)
    history = layer2_model.train(X_balanced, y_balanced, X_val_sequences, y_val_sequences, epochs=50)
    evaluation_results = layer2_model.evaluate_model(X_val_sequences, y_val_sequences)
    print(classification_report(y_val_sequences, np.argmax(layer2_model.model.predict(X_val_sequences), axis=1)))
    layer2_model.plot_training_history(history)

if __name__ == "__main__":
    main()
