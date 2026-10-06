from tensorflow import keras
from keras import layers

def create_3d_cnn_model(input_shape=(16, 64, 64, 3), flow_shape=(16, 64, 64, 2)):
    """
    Create 3D CNN model with dual input streams (spatial and temporal)
    
    Args:
        input_shape: Shape for pixel features (frames, height, width, channels)
        flow_shape: Shape for optical flow features
    
    Returns:
        model: Compiled Keras model
    """
    
    # Spatial stream (pixel features)
    spatial_input = keras.Input(shape=input_shape, name='spatial_input')
    
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(spatial_input)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling3D()(x)
    spatial_features = layers.Dense(256, activation='relu')(x)
    
    # Temporal stream (optical flow)
    temporal_input = keras.Input(shape=flow_shape, name='temporal_input')
    
    y = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(temporal_input)
    y = layers.MaxPooling3D((2, 2, 2))(y)
    y = layers.BatchNormalization()(y)
    
    y = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(y)
    y = layers.MaxPooling3D((2, 2, 2))(y)
    y = layers.BatchNormalization()(y)
    
    y = layers.GlobalAveragePooling3D()(y)
    temporal_features = layers.Dense(256, activation='relu')(y)
    
    # Concatenate features
    combined = layers.concatenate([spatial_features, temporal_features])
    combined = layers.Dropout(0.5)(combined)
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='output')(combined)
    
    # Create model
    model = keras.Model(
        inputs=[spatial_input, temporal_input],
        outputs=output,
        name='violence_detection_3dcnn'
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model
