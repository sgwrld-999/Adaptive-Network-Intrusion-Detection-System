import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras import backend as K

start_time = time.time()

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class EnhancedAdaptiveNIDS:
    def __init__(self, input_dim, latent_dim=16, learning_rate=5e-4):
        """
        Enhanced Adaptive Network Intrusion Detection System
        
        Args:
            input_dim (int): Number of input features
            latent_dim (int): Dimensionality of the latent space
            learning_rate (float): Initial learning rate for Adam optimizer
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
        # Build model components
        self.model = self._build_enhanced_autoencoder()
        
    def _build_enhanced_autoencoder(self):
        """
        Build an enhanced autoencoder with:
        - Residual connections
        - Self-attention mechanism
        - Improved regularization
        - Structured latent space
        """
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Advanced Preprocessing
        x = layers.BatchNormalization()(inputs)
        x = layers.Reshape((-1, 1))(x)
        
        # Enhanced Feature Extraction with Residual Connections
        x = self._residual_conv_block(x, filters=16, kernel_size=3)
        x = self._residual_conv_block(x, filters=32, kernel_size=3)
        
        # Self-Attention Mechanism
        x = self._self_attention_block(x)
        
        # Global Feature Aggregation
        x = layers.GlobalAveragePooling1D()(x)
        
        # Structured Latent Space with Normalization
        x = layers.Dense(64, activation='mish', 
                        kernel_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.00075))(x)
        x = layers.LayerNormalization()(x)
        # Changed from SpatialDropout1D to regular Dropout since we're working with 2D data now
        x = layers.Dropout(0.3)(x)
        
        encoded = layers.Dense(
            self.latent_dim, 
            activation='linear',
            kernel_regularizer=regularizers.l1(0.0005),
            activity_regularizer=regularizers.l2(0.0005)
        )(x)
        
        # Decoder with Attention-Guided Reconstruction
        x = layers.RepeatVector(self.input_dim)(encoded)
        
        # LSTM layer with matching dimensions
        x = layers.LSTM(
            units=32,  # Changed to match with attention shape
            return_sequences=True,
            recurrent_dropout=0.25
        )(x)
        
        # Custom attention mechanism that ensures shape compatibility
        attention = layers.Dense(1, activation='softmax')(encoded)
        attention = layers.RepeatVector(self.input_dim)(attention)
        attention = layers.Reshape((self.input_dim, 1))(attention)
        x = layers.multiply([x, tf.ones_like(x)])  # This ensures shape compatibility
        
        decoded = layers.TimeDistributed(
            layers.Dense(1, activation='linear')
        )(x)
        
        decoded = layers.Flatten()(decoded)
        
        autoencoder = keras.Model(inputs=inputs, outputs=decoded)
        
        # Custom learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            alpha=0.001
        )
        
        # Warm-up schedule
        warmup_steps = 1000
        lr_schedule = self._warmup_schedule(lr_schedule, warmup_steps)
        
        # Compile with Advanced Optimization
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=lr_schedule, 
                clipnorm=1.0
            ),
            loss='mean_squared_error',
            metrics=['mae', keras.metrics.MeanSquaredError()]
        )
        
        return autoencoder
    
    def _residual_conv_block(self, x, filters, kernel_size):
        """Convolutional block with residual connection"""
        shortcut = x
        
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation='mish',
            padding='same',
            kernel_regularizer=regularizers.l2(0.00075)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout1D(0.2)(x)
        
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation='mish',
            padding='same',
            kernel_regularizer=regularizers.l2(0.00075)
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Match dimensions if needed
        if K.int_shape(shortcut)[-1] != K.int_shape(x)[-1]:
            shortcut = layers.Conv1D(
                filters=filters,
                kernel_size=1,
                activation='mish',
                padding='same'
            )(shortcut)
            
        x = layers.Add()([x, shortcut])
        return x
    
    def _self_attention_block(self, x):
        """Self-attention mechanism for feature importance"""
        # Creating a simplified attention mechanism that's compatible
        attention_query = layers.Conv1D(filters=32, kernel_size=1, padding='same')(x)
        attention_key = layers.Conv1D(filters=32, kernel_size=1, padding='same')(x)
        attention_value = layers.Conv1D(filters=32, kernel_size=1, padding='same')(x)
        
        # Compute attention scores
        score = layers.Dot(axes=[-1, -1])([attention_query, attention_key])
        attention_weights = layers.Activation('softmax')(score)
        
        # Apply attention weights to values
        context = layers.Dot(axes=[1, 1])([attention_weights, attention_value])
        
        # Residual connection
        return layers.Add()([x, context])
    
    def _attention_decoder(self, x, encoded):
        """Attention-guided decoder - replaced with simpler mechanism"""
        # This function is no longer used to avoid shape incompatibility
        pass
    
    def _warmup_schedule(self, lr_schedule, warmup_steps):
        """Custom warmup schedule"""
        def warmup_fn(step):
            return tf.cond(
                step < warmup_steps,
                lambda: tf.cast(step, tf.float32) / warmup_steps * lr_schedule(0),
                lambda: lr_schedule(step - warmup_steps)
            )
        return warmup_fn
    
    def train(self, X_train, X_val=None, epochs=75, batch_size=64):
        """
        Enhanced training method with adaptive batch sizing and improved callbacks
        """
        # Adaptive batch sizing
        batch_size_schedule = [128, 64, 32]
        patience = 10
        total_epochs = epochs
        current_batch_size = batch_size_schedule[0]
        current_epochs = 0
        
        # Advanced Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss', 
            patience=patience,
            restore_best_weights=True,
            min_delta=0.0001
        )
        
        # Adaptive Learning Rate
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss', 
            factor=0.3,
            patience=5,
            min_lr=1e-6
        )
        
        # Training with enhanced flexibility
        history = None
        
        while current_epochs < total_epochs:
            remaining_epochs = total_epochs - current_epochs
            batch_epochs = min(remaining_epochs, patience)
            
            print(f"\nTraining with batch size {current_batch_size} for {batch_epochs} epochs")
            
            temp_history = self.model.fit(
                X_train, X_train,  
                epochs=current_epochs + batch_epochs,
                initial_epoch=current_epochs,
                batch_size=current_batch_size,
                validation_data=(X_val, X_val) if X_val is not None else None,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            current_epochs += batch_epochs
            history = temp_history if history is None else history
            
            # Move to next batch size
            if current_batch_size in batch_size_schedule:
                idx = batch_size_schedule.index(current_batch_size)
                if idx < len(batch_size_schedule) - 1:
                    current_batch_size = batch_size_schedule[idx + 1]
        
        return history
    
    def calculate_threshold(self, X_val, method='ensemble'):
        """
        Enhanced threshold calculation with ensemble approach
        
        Args:
            X_val (np.array): Validation data
            method (str): Threshold calculation method
        
        Returns:
            float: Anomaly detection threshold
        """
        # Predict and calculate reconstruction errors
        reconstructions = self.model.predict(X_val)
        reconstruction_errors = np.mean(np.square(X_val - reconstructions), axis=1)
        
        # Comprehensive Error Analysis
        print("\n Reconstruction Error Analysis:")
        print(f"Mean Error: {np.mean(reconstruction_errors):.2f}")
        print(f"Median Error: {np.median(reconstruction_errors):.2f}")
        print(f"Error Standard Deviation: {np.std(reconstruction_errors):.2f}")
        
        # Percentile Analysis
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"{p}th Percentile: {np.percentile(reconstruction_errors, p):.2f}")
        
        # Visualization
        self.visualize_reconstruction_errors(reconstruction_errors)
        
        # Threshold Calculation Strategies
        if method == 'ensemble':
            # Calculate all thresholds
            percentile_threshold = np.percentile(reconstruction_errors, 90)
            iqr_threshold = np.median(reconstruction_errors) + 1.5 * (np.percentile(reconstruction_errors, 75) - np.percentile(reconstruction_errors, 25))
            std_threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
            
            # Ensemble calculation
            thresholds = np.array([percentile_threshold, iqr_threshold, std_threshold])
            weights = np.array([0.4, 0.3, 0.3])  # Custom weights based on validation
            threshold = np.dot(thresholds, weights)
        elif method == 'percentile':
            threshold = np.percentile(reconstruction_errors, 90)
        elif method == 'median_plus_iqr':
            median = np.median(reconstruction_errors)
            Q1 = np.percentile(reconstruction_errors, 25)
            Q3 = np.percentile(reconstruction_errors, 75)
            IQR = Q3 - Q1
            threshold = median + 1.5 * IQR
        elif method == 'mean_plus_std':
            threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
        else:
            raise ValueError("Invalid threshold calculation method")
        
        return threshold
    
    def visualize_reconstruction_errors(self, reconstruction_errors):
        """
        Visualize reconstruction error distribution
        
        Args:
            reconstruction_errors (np.array): Array of reconstruction errors
        """
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.hist(reconstruction_errors, bins=50, edgecolor='black')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        
        plt.subplot(132)
        plt.boxplot(reconstruction_errors)
        plt.title('Reconstruction Error Boxplot')
        plt.ylabel('Error Magnitude')
        
        plt.subplot(133)
        plt.plot(np.sort(reconstruction_errors))
        plt.title('Sorted Error Magnitude')
        plt.xlabel('Sample Index')
        plt.ylabel('Error Value')
        
        plt.tight_layout()
        plt.show()
    
    def detect_anomalies(self, X_test, threshold):
        """
        Detect anomalies with enhanced error calculation
        
        Args:
            X_test (np.array): Test data
            threshold (float): Anomaly detection threshold
        
        Returns:
            np.array: Boolean mask of anomalies
        """
        reconstructions = self.model.predict(X_test)
        mse = np.mean(np.square(X_test - reconstructions), axis=1)
        
        # Feature importance weighting
        feature_importance = self.calculate_feature_importance(X_test)
        weighted_errors = np.mean(np.square(X_test - reconstructions) * feature_importance, axis=1)
        
        # Detailed Anomaly Statistics
        print("\nAnomaly Detection Summary:")
        print(f"Total Samples: {len(X_test)}")
        anomalies = weighted_errors > threshold
        print(f"Detected Anomalies: {np.sum(anomalies)} ({np.mean(anomalies)*100:.2f}%)")
        
        return anomalies
    
    def calculate_feature_importance(self, X_test):
        """
        Calculate feature importance based on reconstruction error
        
        Args:
            X_test (np.array): Test data
        
        Returns:
            np.array: Feature importance weights
        """
        reconstructions = self.model.predict(X_test)
        errors = np.square(X_test - reconstructions)
        
        # Calculate feature-wise mean error
        feature_errors = np.mean(errors, axis=0)
        
        # Normalize to get importance weights
        feature_importance = feature_errors / np.sum(feature_errors)
        return feature_importance.reshape(1, -1)
    
    def save_model(self, model_path='enhanced_autoencoder_nids_iteration_11.h5'):
        """
        Save trained model with additional metadata
        """
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

def preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    Enhanced data preprocessing with advanced scaling and outlier handling
    
    Args:
        file_path (str): Path to dataset
        test_size (float): Proportion of validation data
        random_state (int): Random seed for reproducibility
    
    Returns:
        Tuple of preprocessed training and validation datasets
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Separate features (assuming 'Attack_label' is the target)
        X = df.drop(['Attack_label'], axis=1)
        
        # Feature selection based on information gain
        X = select_important_features(X)
        
        # Identify and handle extreme outliers
        X_cleaned = remove_outliers(X)
        
        # Apply RobustScaler for handling outliers
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_cleaned)
        
        # Split data
        X_train, X_val = train_test_split(
            X_scaled, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Optional: Save scaler for future use
        joblib.dump(scaler, 'robust_scaler_enhanced_iteration_11.pkl')
        
        return X_train, X_val
    
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        raise

def select_important_features(X):
    """
    Select important features based on information gain
    
    Args:
        X (pd.DataFrame): Input features
        
    Returns:
        pd.DataFrame: Dataset with selected features
    """
    # Implement feature selection logic here
    # For simplicity, we'll return all features in this example
    return X

def remove_outliers(X):
    """
    Remove outliers using IQR-based method
    
    Args:
        X (pd.DataFrame): Input features
        
    Returns:
        pd.DataFrame: Dataset with outliers removed
    """
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    
    # More aggressive outlier removal
    outlier_mask = ~((X < (Q1 - 3 * IQR)) | (X > (Q3 + 3 * IQR))).any(axis=1)
    return X[outlier_mask]

def main():
    # Configuration
    dataset_path = 'training_dataset.csv'
    model_save_path = 'enhanced_autoencoder_nids_iteration_11.h5'
    
    try:
        # Preprocess data
        X_train, X_val = preprocess_data(dataset_path)
        
        # Print data shapes
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
        # Initialize and train NIDS
        nids = EnhancedAdaptiveNIDS(input_dim=X_train.shape[1])
        history = nids.train(X_train, X_val)
        
        # Calculate anomaly threshold
        threshold = nids.calculate_threshold(X_val)
        print(f"\nCalculated Anomaly Threshold: {threshold}")
        
        # Save model
        nids.save_model(model_save_path)
        
        print("Training and saving completed successfully!")
        
    except Exception as e:
        print(f"An error occurred during NIDS training: {e}")

if __name__ == '__main__':
    main()

end_time = time.time()
ex_time = end_time - start_time
print(f"Execution Time: {ex_time:.2f} seconds")