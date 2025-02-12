import pandas as pd
import numpy as np
import time
import os
from collections import Counter
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import GPUtil
import psutil

class ResourceManager:
    def __init__(self, cpu_limit_percent=10, memory_limit_percent=10):
        self.total_cores = 64
        self.total_memory = 503.03  # GB
        self.available_gpus = [2, 3, 4, 5, 7]
        
        # Calculate resource limits (10% of total)
        self.max_cores = max(1, int(self.total_cores * (cpu_limit_percent / 100)))
        self.max_memory = int(self.total_memory * (memory_limit_percent / 100))
        self.chunk_size = min(5000, int((self.max_memory * 1024 * 1024 * 1024) / (8 * 1000)))
        
        # Set system parameters
        torch.set_num_threads(self.max_cores)
        os.environ["OMP_NUM_THREADS"] = str(self.max_cores)
    
    def get_optimal_gpu(self):
        return 2  # Using GPU 3 (Tesla V100) with low usage
    
    def get_batch_size(self):
        return min(2000, self.chunk_size)

class OptimizedDataProcessor:
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.selected_features = [
            'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
            'Rate', 'Drate', 'syn_flag_number', 'ack_flag_number',
            'TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'IAT',
            'Magnitue', 'Weight', 'IPv', 'label'
        ]
        
    def load_and_process_file(self, file_path):
        chunks = []
        chunk_size = self.resource_manager.chunk_size
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk = chunk[self.selected_features]
            chunks.append(chunk)
            
            if len(chunks) % 5 == 0:
                torch.cuda.empty_cache()
                
        return pd.concat(chunks, ignore_index=True)
    
    def balance_classes(self, X, y):
        """Balance classes using undersampling for majority and replication for minority"""
        class_counts = Counter(y)
        median_count = np.median(list(class_counts.values()))
        target_count = int(median_count)  # Use median as target count
        
        balanced_X = []
        balanced_y = []
        
        for class_label in class_counts:
            mask = y == class_label
            X_class = X[mask]
            y_class = y[mask]
            
            if len(X_class) > target_count:
                # Undersample majority classes
                indices = np.random.choice(len(X_class), target_count, replace=False)
                X_balanced = X_class[indices]
                y_balanced = y_class[indices]
            else:
                # Oversample minority classes using simple replication
                indices = np.random.choice(len(X_class), target_count, replace=True)
                X_balanced = X_class[indices]
                y_balanced = y_class[indices]
            
            balanced_X.append(X_balanced)
            balanced_y.append(y_balanced)
        
        return np.vstack(balanced_X), np.concatenate(balanced_y)
    
    def process_data(self, df_chunk, device):
        """Process a single chunk of data"""
        X = df_chunk.drop('label', axis=1)
        y = df_chunk['label']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Move to GPU in smaller batches
        batch_size = self.resource_manager.get_batch_size()
        processed_chunks = []
        
        for i in range(0, len(X_scaled), batch_size):
            batch = X_scaled[i:min(i + batch_size, len(X_scaled))]
            tensor_batch = torch.tensor(batch, dtype=torch.float32, device=device)
            processed = self._gpu_operations(tensor_batch)
            processed_chunks.append(processed.cpu().numpy())
            torch.cuda.empty_cache()
        
        X_processed = np.vstack(processed_chunks)
        
        # Balance classes
        X_balanced, y_balanced = self.balance_classes(X_processed, y)
        
        return X_balanced, y_balanced
    
    def _gpu_operations(self, tensor):
        # Add any GPU-specific operations here
        # For example: normalization, feature transformation, etc.
        return tensor

def main():
    start_time = time.time()
    
    # Initialize resource manager with 10% limit
    resource_manager = ResourceManager(cpu_limit_percent=10, memory_limit_percent=10)
    processor = OptimizedDataProcessor(resource_manager)
    
    # Set up GPU
    gpu_id = resource_manager.get_optimal_gpu()
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    file_paths = [
        "part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv",
        "part-00001-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv",
        "part-00002-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv",
        "part-00003-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv",
        "part-00004-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
    ]
    
    # Process files one at a time
    for idx, file_path in enumerate(file_paths):
        print(f"Processing file {idx + 1}/{len(file_paths)}")
        
        # Load and process current file
        df_chunk = processor.load_and_process_file(file_path)
        X_processed, y_processed = processor.process_data(df_chunk, device)
        
        # Process in smaller chunks
        chunk_size = resource_manager.chunk_size
        for i in range(0, len(X_processed), chunk_size):
            end_idx = min(i + chunk_size, len(X_processed))
            X_chunk = X_processed[i:end_idx]
            y_chunk = y_processed[i:end_idx]
            
            # Create and save DataFrame
            chunk_df = pd.DataFrame(
                X_chunk,
                columns=[f'feature_{j}' for j in range(X_chunk.shape[1])]
            )
            chunk_df['label'] = y_chunk
            
            # Save to parquet with compression
            output_path = f'processed_file_{idx}_chunk_{i//chunk_size}.parquet'
            chunk_df.to_parquet(
                output_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            # Clear memory
            del chunk_df
            torch.cuda.empty_cache()
    
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()