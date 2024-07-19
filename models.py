# TCN NET
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.layers import WeightNormalization
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow.keras.models import Model
import keras.backend as K
from tensorflow.keras.constraints import max_norm

# Model
def residual_block(x, filters = 128, kernel_size = 3, dilation_rate = 1):
    # Residual
    x_res = x
    # ConvNet
    x_out = WeightNormalization(layers.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, padding = 'causal'))(x)
    x_out = layers.Activation('elu')(x_out)
    x_out = WeightNormalization(layers.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, padding = 'causal'))(x_out)
    x_out = layers.Activation('elu')(x_out)
    # Residual Outputs
    residual = layers.Add()([x_res, x_out])
    return residual, x_out
def Detection_models(input_len = 2000, n_channel = 1, n_filters = 128):
    inputs = layers.Input(shape = (input_len, n_channel))
    resampled = layers.Conv1D(filters = 32, kernel_size = 7, strides = 2, padding = 'same')(inputs)
    # TCN Net
    tcn = layers.Conv1D(filters = n_filters, kernel_size = 1, padding = 'same')(resampled)
    tcn, s1 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 1)
    tcn, s2 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 2)
    tcn, s3 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 4)
    tcn, s4 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 8)
    tcn, s5 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 16)
    tcn, s6 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 32)
    tcn, s7 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 64)
    tcn, s8 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 128)
    tcn, s9 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 256)
    tcn, s10 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 512)
    # Outputs
    skip_connection = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
    skip_connection = layers.Add()(skip_connection)
    skip_connection = layers.Activation('elu')(skip_connection)
    outputs = layers.GlobalAveragePooling1D()(skip_connection)
    # Classifier
    outputs = layers.Dense(n_filters, activation = 'elu')(outputs)
    outputs = layers.Dense(n_filters, activation = 'elu')(outputs)
    outputs = layers.Dense(n_filters, activation = 'elu')(outputs)
    outputs = layers.Dense(1, activation = 'sigmoid')(outputs)
    # Model compile
    model = Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam( learning_rate = 0.0001 ) # 0.0001
    rocauc = tf.keras.metrics.AUC(curve = 'ROC', name = 'rocauc')
    losses = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer = optimizer, loss = losses, metrics = [rocauc, 'acc'])
    return model