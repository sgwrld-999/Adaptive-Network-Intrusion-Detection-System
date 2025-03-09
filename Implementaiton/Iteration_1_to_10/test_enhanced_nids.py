import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from enhanced_nids import EnhancedAdaptiveNIDS

def load_and_preprocess_test_data(test_file_path, scaler_path='robust_scaler_enhanced.pkl'):
    """
    Load and preprocess test data using the previously fitted scaler
    
    Args:
        test_file_path (str): Path to test dataset
        scaler_path (str): Path to saved scaler
    
    Returns:
        tuple: X_test (features), y_test (labels)
    """
    # Load test data
    df_test = pd.read_csv(test_file_path)
    
    # Separate features and labels
    X_test = df_test.drop(['Attack_label'], axis=1)
    y_test = df_test['Attack_label']
    
    # Load saved scaler
    scaler = joblib.load(scaler_path)
    
    # Apply same preprocessing as training data
    X_test_scaled = scaler.transform(X_test)
    
    return X_test_scaled, y_test

def evaluate_model(model, X_test, y_test, threshold):
    """
    Evaluate the model's performance on test data
    
    Args:
        model (EnhancedAdaptiveNIDS): Trained model
        X_test (np.array): Test features
        y_test (np.array): Test labels (0=normal, 1=attack)
        threshold (float): Anomaly detection threshold
    
    Returns:
        dict: Performance metrics
    """
    # Detect anomalies
    y_pred = model.detect_anomalies(X_test, threshold)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # Calculate feature importances
    feature_importance = model.calculate_feature_importance(X_test)
    
    # Return metrics
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'feature_importance': feature_importance
    }

def visualize_feature_importance(feature_importance, feature_names):
    """
    Visualize feature importance for interpretability
    
    Args:
        feature_importance (np.array): Feature importance weights
        feature_names (list): Names of features
    """
    # Get top features
    importance = feature_importance.flatten()
    indices = np.argsort(importance)[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance for Anomaly Detection')
    plt.bar(range(min(20, len(indices))), 
            importance[indices[:20]], 
            align='center')
    plt.xticks(range(min(20, len(indices))), 
              [feature_names[i] for i in indices[:20]], 
              rotation=90)
    plt.tight_layout()
    plt.show()

def monte_carlo_uncertainty(model, X_test, n_iterations=10):
    """
    Perform Monte Carlo dropout for uncertainty estimation
    
    Args:
        model (EnhancedAdaptiveNIDS): Trained model
        X_test (np.array): Test data
        n_iterations (int): Number of forward passes
    
    Returns:
        tuple: Mean predictions, prediction variance (uncertainty)
    """
    # Set model to train mode to enable dropout during inference
    model.model.layers[0].trainable = True
    
    # Perform multiple forward passes
    predictions = []
    for _ in range(n_iterations):
        reconstructions = model.model.predict(X_test)
        errors = np.mean(np.square(X_test - reconstructions), axis=1)
        predictions.append(errors)
    
    # Reset model to inference mode
    model.model.layers[0].trainable = False
    
    # Calculate statistics
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    uncertainty = np.var(predictions, axis=0)
    
    return mean_pred, uncertainty

def main():
    # Configuration
    test_file_path = 'testing_dataset.csv'
    model_path = 'enhanced_autoencoder_nids.h5'
    threshold_path = 'anomaly_threshold.pkl'
    
    try:
        # Start timer
        start_time = time.time()
        
        # Load test data
        X_test, y_test = load_and_preprocess_test_data(test_file_path)
        
        # Load model and threshold
        model = EnhancedAdaptiveNIDS(input_dim=X_test.shape[1])
        model.model = tf.keras.models.load_model(model_path)
        threshold = joblib.load(threshold_path)
        
        print("Model and data loaded successfully!")
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, threshold)
        
        # Print results
        print("\nModel Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Get feature names
        feature_names = pd.read_csv(test_file_path).drop(['Attack_label'], axis=1).columns.tolist()
        
        # Visualize feature importance
        visualize_feature_importance(metrics['feature_importance'], feature_names)
        
        # Uncertainty estimation
        mean_pred, uncertainty = monte_carlo_uncertainty(model, X_test)
        
        # Visualize uncertainty
        plt.figure(figsize=(10, 6))
        plt.hist(uncertainty, bins=50)
        plt.title('Prediction Uncertainty Distribution')
        plt.xlabel('Uncertainty (Variance)')
        plt.ylabel('Frequency')
        plt.show()
        
        # Execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
    except Exception as e:
        print(f"An error occurred during testing: {e}")

if __name__ == "__main__":
    main() 