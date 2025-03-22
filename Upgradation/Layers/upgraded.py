import tensorflow as tf
import datetime
from ANIDS import EnhancedAdaptiveNIDS, AdaptiveNIDSLayer2

# Define input dimensions
input_dim = 50  # Adjust based on dataset
latent_dim = 64
seq_length = 10
num_classes = 5  # Adjust based on dataset

# Initialize Layer 1 and Layer 2
layer1 = EnhancedAdaptiveNIDS(input_dim=input_dim, latent_dim=latent_dim)
layer2 = AdaptiveNIDSLayer2(input_dim=input_dim, num_classes=num_classes, seq_length=seq_length)

# TensorBoard log directory
log_dir = "logs/architecture/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# Create a dummy input tensor for visualization
dummy_input_l1 = tf.random.normal((1, input_dim))  # Shape for Autoencoder
dummy_input_l2 = tf.random.normal((1, seq_length, input_dim))  # Shape for CNN-BiLSTM

# Log models to TensorBoard
with writer.as_default():
    tf.summary.trace_on(graph=True)  # Enable tracing for graph visualization
    _ = layer1.model(dummy_input_l1)  # Run model once to create graph
    _ = layer2.model(dummy_input_l2)  # Run model once to create graph
    tf.summary.trace_export(name="Model_Graph", step=0)  # Export the graph
    writer.flush()  # Ensure logs are properly written

print(f"TensorBoard logs saved to: {log_dir}")
print("Run the following command to view the model: tensorboard --logdir=logs/architecture")
