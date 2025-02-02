import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, Conv1D, Conv2D, Conv3D,
    LSTM, GRU, Bidirectional, Flatten, MaxPooling1D, MaxPooling2D,
    GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization,
    Add, MultiHeadAttention, LayerNormalization, Reshape, TimeDistributed,
    SeparableConv1D, DepthwiseConv2D, Lambda, Multiply, Activation
)
from tensorflow.keras.regularizers import l1_l2
import numpy as np
import os
from keras.models import Sequential
from datetime import datetime
from sklearn.metrics import roc_auc_score
import pandas as pd
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Flatten, Multiply, GlobalAveragePooling1D, Softmax, Reshape
from tensorflow.keras import Model
 
from tensorflow.keras import backend as K  # Add this import
import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, SimpleRNN, LSTM, GRU, 
    Bidirectional, Flatten, MaxPooling1D, BatchNormalization,
    Add, Multiply, Activation, Reshape, Concatenate
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from keras import layers
class DeepLearningModels:
    def __init__(self, input_shapes, learning_rate=0.001):
        self.input_shapes = input_shapes
        self.learning_rate = learning_rate
        
import numpy as np
import tensorflow as tf
import random
 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed()
 
 

class DeepLearningModels:
    def __init__(self, input_shapes, learning_rate=0.001):
        """
        Initialize with dictionary of input shapes for data
        input_shapes = {
            'main': (features,)  # Single input shape for features
        }
        """
        self.input_shapes = input_shapes
        self.learning_rate = learning_rate
        
        # Update which models need reshaping for (Features, 1) format
        self.models_needing_cnn_format = {
            'MultiScaleCNN', 
            'DilatedCNN',
            'PyramidNet',
            'HybridResidualAttention',
            'MultiScaleFeatureFusion',
            'DeepChannelNet',
            'CascadedNet',
            'TemporalConvolutionalNet',
            'DenseNet',
            'simpleBuildCnnModel',
            'simpleBuild1dCnnModel'
        }
 

 
        # Models needing (1, Features) format for LSTM/Transformer
        self.models_needing_3d = {
            'HybridTransformerLSTM',
            'DenseTransformer',
            'GatedRecurrentMixer',
            'DeepBiGRU',
            'AttentionLSTM',
            'HierarchicalAttentionNet',
            'SqueezeExciteNet',
            'FTTransformer',
            'GraphAttentionNetwork',
            'GraphConvolutionalNetwork',
            'NestedLSTM',
            'NeuralTuringMachine',
            'simpleBuildRnnModel',
            'simpleBuildLstmModel',
            'simpleBuildGruModel',
            'simpleBuildBidirectionalLstmModel'
        }


    def _get_input_shape(self, model_name):
        """Determine the appropriate input shape based on the model type"""
        base_shape = self.input_shapes['main']  # (features,)
        
        if model_name in self.models_needing_cnn_format:
            # For CNN models: (features, 1)
            return base_shape + (1,)
        elif model_name in self.models_needing_3d:
            # For LSTM/Transformer models: (1, features)
            return (1,) + base_shape
        return base_shape

    # Updated MultiScaleCNN for (Features, 1) input
    def _create_multiscale_cnn(self, input_shape):
        """Create a multi-scale CNN with parallel pathways"""
        inputs = Input(shape=input_shape, name="input_main")
        print(f"MultiScaleCNN input shape: {input_shape}")
        
        # Multiple parallel convolution paths with different kernel sizes
        conv_paths = []
        for kernel_size in [3, 5, 7]:
            path = Conv1D(32, kernel_size, padding='same', activation='relu')(inputs)
            path = BatchNormalization()(path)
            conv_paths.append(path)
        
        # Concatenate paths along the channel dimension (-1)
        x = Concatenate(axis=-1)(conv_paths)
        
        # Additional convolution
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    # Similarly update other CNN-based models...
    def _create_dilated_cnn(self, input_shape):
        """Create a CNN with dilated convolutions"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Multiple dilated convolution paths
        paths = []
        for dilation_rate in [1, 2, 4]:
            x = Conv1D(32, 3, dilation_rate=dilation_rate, 
                      padding='same', activation='relu')(inputs)
            x = BatchNormalization()(x)
            paths.append(x)
        
        x = Concatenate(axis=-1)(paths)
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)


    def create_model(self, model_name):
        """Factory method to create various deep learning models"""
        # Update input shapes based on model requirements
        input_shape = self._get_input_shape(model_name)
        
        model_creators = {
            'DeepResDenseNet': lambda: self._create_deep_residual_dense_net(input_shape),
            'MultiScaleCNN': lambda: self._create_multiscale_cnn(input_shape),
            'HybridTransformerLSTM': lambda: self._create_hybrid_transformer_lstm(input_shape),
            'DenseTransformer': lambda: self._create_dense_transformer(input_shape),
            'DeepBiGRU': lambda: self._create_deep_bigru(input_shape),
            'PyramidNet': lambda: self._create_pyramid_net(input_shape),
            'MultiPathResNet': lambda: self._create_multipath_resnet(input_shape),
            'AttentionLSTM': lambda: self._create_attention_lstm(input_shape),
            'DilatedCNN': lambda: self._create_dilated_cnn(input_shape),
            'HierarchicalAttentionNet': lambda: self._create_hierarchical_attention_net(input_shape),
            'CrossStitchNet': lambda: self._create_cross_stitch_net(input_shape),
            'GatedRecurrentMixer': lambda: self._create_gated_recurrent_mixer(input_shape),
            'MultiModalFusion': lambda: self._create_multimodal_fusion(input_shape),
            'DeepChannelNet': lambda: self._create_deep_channel_net(input_shape),
            'SqueezeExciteNet': lambda: self._create_squeeze_excite_net(input_shape),
            'DualPathNetwork': lambda: self._create_dual_path_network(input_shape),
            'CascadedNet': lambda: self._create_cascaded_net(input_shape),
            'DeepInteractionNet': lambda: self._create_deep_interaction_net(input_shape),
            'HybridResidualAttention': lambda: self._create_hybrid_residual_attention(input_shape),
            'MultiScaleFeatureFusion': lambda: self._create_multiscale_feature_fusion(input_shape),
            'TabNet': lambda: self._create_tabnet(input_shape),
            'DeepFM': lambda: self._create_deepfm(input_shape),
            'TabTransformer': lambda: self._create_tabtransformer(input_shape),
            'NODE': lambda: self._create_node(input_shape),
            'EntityEmbeddings': lambda: self._create_entity_embeddings(input_shape),
            'FTTransformer': lambda: self._create_ft_transformer(input_shape),
            'WideAndDeep': lambda: self._create_wide_and_deep(input_shape),
            'AutoInt': lambda: self._create_autoint(input_shape),
            'DCN': lambda: self._create_dcn(input_shape),
            'CatNet': lambda: self._create_catnet(input_shape),
            'DeepGBM': lambda: self._create_deepgbm(input_shape),
            'NeuralDecisionForest': lambda: self._create_neural_decision_forest(input_shape),
            'NetDNF': lambda: self._create_net_dnf(input_shape),
            'GraphAttentionNetwork': lambda: self._create_graph_attention_network(input_shape),
            'VariationalAutoencoder': lambda: self._create_variational_autoencoder(input_shape),
            'GraphConvolutionalNetwork': lambda: self._create_graph_convolutional_network(input_shape),
            'CapsuleNetwork': lambda: self._create_capsule_network(input_shape),
            'DenseAutoencoder': lambda: self._create_dense_autoencoder(input_shape),
            'NeuralTuringMachine': lambda: self._create_neural_turing_machine(input_shape),
            'HighwayNetwork': lambda: self._create_highway_network(input_shape),
            'DenseNet': lambda: self._create_densenet(input_shape),
            'DeepBeliefNetwork': lambda: self._create_deep_belief_network(input_shape),
            'NestedLSTM': lambda: self._create_nested_lstm(input_shape),
            'TemporalConvolutionalNet': lambda: self._create_temporal_convolutional_net(input_shape),
            'NeuralArchitectureSearch': lambda: self._create_neural_architecture_search(input_shape),
            'simpleBuildFnnModel': lambda: self._simple_build_fnn_model(input_shape),
            'simpleBuildCnnModel': lambda: self._simple_build_cnn_model(input_shape),
            'simpleBuildRnnModel': lambda: self._simple_build_rnn_model(input_shape),
            'simpleBuildLstmModel': lambda: self._simple_build_lstm_model(input_shape),
            'simpleBuildGruModel': lambda: self._simple_build_gru_model(input_shape),
            'simpleBuildBidirectionalLstmModel': lambda: self._simple_build_bidirectional_lstm_model(input_shape),
            'simpleBuild1dCnnModel': lambda: self._simple_build_1d_cnn_model(input_shape),
            'simpleBuildMlpModel': lambda: self._simple_build_mlp_model(input_shape),
            'simpleBuildResnetModel': lambda: self._simple_build_resnet_model(input_shape)


     
        }
        
        if model_name not in model_creators:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_creators[model_name]()

    def _create_transformer_block(self, inputs, num_heads, key_dim, dropout_rate=0.1):
        """Create a transformer block with multi-head attention."""
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )(inputs, inputs)
        attention_output = Dropout(dropout_rate)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        ffn_output = Dense(key_dim * 4, activation='relu')(attention_output)
        ffn_output = Dense(inputs.shape[-1])(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        
        return LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)

    def _create_hybrid_transformer_lstm(self, input_shape):
        """Create a hybrid model combining transformer and LSTM architectures."""
        inputs = Input(shape=input_shape, name="input_main")
        print(inputs.shape)
        # LSTM layers
        x = LSTM(64, return_sequences=True)(inputs)
        x = Bidirectional(LSTM(32))(x)
        
        # Dense layers for classification
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_deep_residual_dense_net(self, input_shape):
        """Create a deep residual dense network with skip connections"""
        inputs = Input(shape=input_shape, name="input_main")
        
        x = inputs
        # Create dense blocks with residual connections
        for units in [128, 256, 128, 64]:
            residual = x
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            
            # Add residual if shapes match, otherwise transform
            if residual.shape[-1] != units:
                residual = Dense(units)(residual)
            x = Add()([x, residual])
        
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_cross_stitch_net(self, input_shape):
        """Create a cross-stitch network for feature sharing between streams"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Initialize multiple streams
        num_streams = 3
        streams = []
        for i in range(num_streams):
            stream = Dense(64, activation='relu', name=f'stream_{i}_initial')(inputs)
            streams.append(stream)
        
        # Cross-stitch blocks
        for block_idx in range(3):
            cross_stitch_outputs = []
            
            # Create a cross-stitch block
            for i in range(num_streams):
                # Linear combination of all streams
                combined = []
                for j, stream in enumerate(streams):
                    # Apply learnable scaling factor to each stream
                    scale = Dense(64, use_bias=False, 
                                name=f'cross_stitch_scale_{block_idx}_{i}_{j}')(stream)
                    combined.append(scale)
                
                # Sum all scaled streams
                merged = Add(name=f'cross_stitch_merge_{block_idx}_{i}')(combined)
                
                # Apply non-linearity
                output = Dense(64, activation='relu',
                            name=f'cross_stitch_output_{block_idx}_{i}')(merged)
                cross_stitch_outputs.append(output)
            
            # Update streams for next block
            streams = cross_stitch_outputs
        
        # Combine final streams
        x = Concatenate()(streams)
        
        # Final classification layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        print("\nCrossStitchNet Model Summary:")
        model.summary()
        
        return model    

    def _create_dense_transformer(self, input_shape):
        """Create a transformer-based model with dense connections"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Multiple transformer blocks with dense connections
        transformer_outputs = []
        x = inputs
        
        for i in range(3):
            transformer_block = self._create_transformer_block(x, num_heads=4, key_dim=64)
            transformer_outputs.append(transformer_block)
            if i < 2:  # Don't concatenate after last block
                x = Concatenate(axis=-1)([x, transformer_block])
        
        x = GlobalAveragePooling1D()(transformer_outputs[-1])
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_deep_bigru(self, input_shape):
        """Create a deep bidirectional GRU network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        x = inputs
        # Multiple stacked Bidirectional GRU layers
        for units in [64, 32, 16]:
            x = Bidirectional(GRU(units, return_sequences=True))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
        
        x = Bidirectional(GRU(16))(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_pyramid_net(self, input_shape):
        """Create a pyramid network with gradually increasing feature maps"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Pyramid of increasing feature maps
        x = Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        skip1 = x
        
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = Conv1D(256, 3, activation='relu', padding='same')(x)
        
        # Upsampling path
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = Concatenate()([x, skip1])
        
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_multipath_resnet(self, input_shape):
        """Create a multi-path residual network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Multiple parallel residual paths
        paths = []
        for units in [64, 128, 256]:
            x = inputs
            for _ in range(2):
                residual = x
                x = Dense(units, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dense(units)(x)
                
                if residual.shape[-1] != units:
                    residual = Dense(units)(residual)
                x = Add()([x, residual])
                x = tf.keras.layers.ReLU()(x)
            
            paths.append(x)
        
        x = Concatenate()(paths)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_attention_lstm(self, input_shape):
        """Create an LSTM model with self-attention mechanism"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # LSTM with attention
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        
        # Self-attention mechanism
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        
        # Replace tf.nn.softmax with Keras Softmax layer
        attention = Softmax()(attention)
        
        # Replace tf.expand_dims with Reshape to expand dimensions
        attention = Reshape((-1, 1))(attention)
        
        # Apply attention weights
        x = Multiply()([x, attention])
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)
  
    def _create_hierarchical_attention_net(self, input_shape):
            """Create a hierarchical attention network"""
            inputs = Input(shape=input_shape, name="input_main")
            
            # Local attention
            local_attention = MultiHeadAttention(
                num_heads=4, key_dim=32
            )(inputs, inputs)
            local_attention = LayerNormalization()(local_attention + inputs)
            
            # Feature extraction
            x = Conv1D(64, 3, activation='relu', padding='same')(local_attention)
            x = BatchNormalization()(x)
            
            # Global attention
            global_attention = MultiHeadAttention(
                num_heads=2, key_dim=64
            )(x, x)
            global_attention = LayerNormalization()(global_attention + x)
            
            x = GlobalAveragePooling1D()(global_attention)
            x = Dense(128, activation='relu')(x)
            output = Dense(1, activation='sigmoid')(x)
            
            return Model(inputs=inputs, outputs=output)

    def _create_gated_recurrent_mixer(self, input_shape):
        """Create a gated recurrent mixer with adaptive feature fusion"""
        inputs = Input(shape=input_shape, name="input_main")
        print(f"GatedRecurrentMixer input shape: {input_shape}")
        
        # Define common units for all paths to ensure shape compatibility
        hidden_units = 64
        
        # GRU path with fixed output shape
        gru_path = Bidirectional(
            GRU(hidden_units // 2, return_sequences=True),  # Use half units for bidirectional
            merge_mode='concat'
        )(inputs)
        print(f"GRU path shape: {gru_path.shape}")
        
        # CNN path with matching output shape
        cnn_path = Conv1D(hidden_units, 3, padding='same', activation='relu')(inputs)
        cnn_path = BatchNormalization()(cnn_path)
        print(f"CNN path shape: {cnn_path.shape}")
        
        # Gating mechanism with matching shape
        gate = Conv1D(hidden_units, 1, activation='sigmoid')(inputs)
        print(f"Gate shape: {gate.shape}")
        
        # Create gated combination
        gated_gru = Multiply()([gru_path, gate])
        gated_cnn = Multiply()([cnn_path, Lambda(lambda x: 1 - x)(gate)])
        
        # Combine paths
        x = Add()([gated_gru, gated_cnn])
        print(f"Combined shape: {x.shape}")
        
        # Final processing
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        print("\nGatedRecurrentMixer Model Summary:")
        model.summary()
        
        return model

    def _create_multimodal_fusion(self, input_shape):
        """Create a multimodal fusion network with adaptive weighting"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Deep feature extraction
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        
        # Importance weighting
        attention = Dense(1, activation='sigmoid')(x)
        x = Multiply()([x, attention])
        
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_deep_channel_net(self, input_shape):
        """Create a deep channel network with feature grouping."""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Channel-wise processing
        groups = []
        num_groups = 4
        features_per_group = input_shape[-1] // num_groups

        for i in range(num_groups):
            start_idx = i * features_per_group
            # Handle the last group to include any remainder features
            end_idx = (i + 1) * features_per_group if i < num_groups - 1 else input_shape[-1]
            
            # Ensure slicing produces valid inputs
            if start_idx < end_idx:
                group = Lambda(lambda x: x[..., start_idx:end_idx])(inputs)
                x = Conv1D(32, 3, activation='relu', padding='same')(group)
                x = BatchNormalization()(x)
                groups.append(x)

        # Check if any groups were created
        if not groups:
            raise ValueError("Feature grouping resulted in no valid groups. Check input shape and number of groups.")
        
        x = Concatenate()(groups)
        x = Conv1D(128, 3, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)
    
    
    
    def _create_squeeze_excite_net(self, input_shape):
        """Create a squeeze-and-excitation network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        
        # Squeeze and Excitation block
        se = GlobalAveragePooling1D()(x)
        se = Dense(64 // 16, activation='relu')(se)
        se = Dense(64, activation='sigmoid')(se)
        se = Reshape((1, 64))(se)
        
        x = Multiply()([x, se])
        x = Conv1D(128, 3, activation='relu', padding='same')(x)  # Fixed padding
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)


    def _create_dual_path_network(self, input_shape):
        """Create a dual path network with residual and dense connections"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Residual path
        res_path = inputs
        for units in [64, 128, 64]:
            residual = res_path
            res_path = Dense(units, activation='relu')(res_path)
            if residual.shape[-1] != units:
                residual = Dense(units)(residual)
            res_path = Add()([res_path, residual])
        
        # Dense path
        dense_path = inputs
        dense_features = [dense_path]
        for units in [64, 128, 64]:
            dense_path = Dense(units, activation='relu')(Concatenate()(dense_features))
            dense_features.append(dense_path)
        
        # Combine paths
        x = Concatenate()([res_path, dense_path])
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_cascaded_net(self, input_shape):
        """Create a cascaded network with progressive feature refinement"""
        inputs = Input(shape=input_shape, name="input_main")
        
        features = []
        x = inputs
        
        # Progressive feature extraction
        for filters in [32, 64, 128]:
            x = Conv1D(filters, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            features.append(GlobalAveragePooling1D()(x))
        
        # Cascade features
        cascade = features[0]
        for feature in features[1:]:
            cascade = Concatenate()([cascade, feature])
            cascade = Dense(128, activation='relu')(cascade)
        
        x = Dense(256, activation='relu')(cascade)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_deep_interaction_net(self, input_shape):
        """Create a deep interaction network with cross-feature learning"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Deep feature extraction
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        
        # Self-interaction layer
        interaction = Dense(32, activation='relu')(x)
        
        # Combine features
        x = Concatenate()([x, interaction])
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_hybrid_residual_attention(self, input_shape):
        """Create a hybrid residual attention network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Main branch
        main = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        main = BatchNormalization()(main)
        
        # Attention branch
        attention = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        attention = BatchNormalization()(attention)
        attention = Conv1D(1, 1, activation='sigmoid')(attention)
        
        # Apply attention and residual
        x = Multiply()([main, attention])
        x = Add()([x, main])  # Residual connection
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_multiscale_feature_fusion(self, input_shape):
        """Create a multi-scale feature fusion network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Multi-scale feature extraction
        scales = []
        for pool_size in [2, 4, 8]:
            x = Conv1D(32, pool_size, activation='relu', padding='same')(inputs)
            x = MaxPooling1D(pool_size)(x)
            x = Conv1D(64, 3, activation='relu', padding='same')(x)
            x = GlobalAveragePooling1D()(x)
            scales.append(x)
        
        # Feature fusion
        x = Concatenate()(scales)
        x = Dense(128, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_tabnet(self, input_shape):
        """Create a TabNet architecture with attention-based feature selection"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Feature transformer
        x = Dense(256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        
        # Decision steps with attention
        steps = []
        for _ in range(3):  # Number of decision steps
            # Feature selection
            attention = Dense(input_shape[0], activation='sigmoid')(x)
            selected = Multiply()([inputs, attention])
            
            # Feature processing
            step = Dense(128, activation='relu')(selected)
            step = BatchNormalization()(step)
            steps.append(step)
            
            # Residual update
            x = Add()([Dense(128)(x), step])
        
        x = Concatenate()(steps)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_deepfm(self, input_shape):
        """Create a Deep Factorization Machine network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # FM Component
        fm_linear = Dense(1, use_bias=True)(inputs)
        
        # Second-order interactions
        factor_dim = 16
        v = Dense(factor_dim)(inputs)
        square_of_sum = Lambda(lambda x: K.square(K.sum(x, axis=1)), output_shape=(factor_dim,))(v)
        sum_of_square = Lambda(lambda x: K.sum(K.square(x), axis=1), output_shape=(factor_dim,))(v)
        fm_interactions = Lambda(lambda x: 0.5 * K.reshape(x[0] - x[1], (-1, 1)), output_shape=(1,))( [square_of_sum, sum_of_square])
        
        # Deep Component
        deep = Dense(256, activation='relu')(inputs)
        deep = BatchNormalization()(deep)
        deep = Dense(128, activation='relu')(deep)
        deep = Dense(64, activation='relu')(deep)
        deep = Dense(1, activation='linear')(deep)
        
        # Combine outputs
        output = Add()([fm_linear, fm_interactions, deep])
        output = Activation('sigmoid')(output)
        
        return Model(inputs=inputs, outputs=output)

 

    def _create_node(self, input_shape):
        """Create Neural Oblivious Decision Ensembles"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Create multiple decision trees
        trees = []
        for _ in range(8):  # Number of trees
            # Create decision nodes
            splits = Dense(input_shape[0], activation='sigmoid')(inputs)
            features = Multiply()([inputs, splits])
            
            # Tree processing
            tree = Dense(64, activation='relu')(features)
            tree = BatchNormalization()(tree)
            trees.append(tree)
        
        # Ensemble combination
        x = Add()(trees)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_entity_embeddings(self, input_shape):
        """Create network with entity embeddings for categorical variables"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Embedding layer for features
        embedding_dim = 16
        x = Dense(embedding_dim)(inputs)
        
        # Process embeddings
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_ft_transformer(self, input_shape):
        """Create Feature Tokenizer Transformer"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Feature tokenization
        x = Dense(64)(inputs)  # Token embedding
        
        # Transformer blocks
        for _ in range(3):
            # Multi-head attention
            att = MultiHeadAttention(num_heads=8, key_dim=8)(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + att)
            
            # Position-wise FFN
            ffn = Dense(128, activation='relu')(x)
            ffn = Dense(64)(ffn)
            x = LayerNormalization(epsilon=1e-6)(x + ffn)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)



    def _create_wide_and_deep(self, input_shape):
        """Create Wide & Deep network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Wide path
        wide = Dense(256)(inputs)
        wide = Dense(1, use_bias=False)(wide)
        
        # Deep path
        deep = Dense(256, activation='relu')(inputs)
        deep = BatchNormalization()(deep)
        deep = Dense(128, activation='relu')(deep)
        deep = Dense(64, activation='relu')(deep)
        deep = Dense(1)(deep)
        
        # Combine paths
        combined = Add()([wide, deep])
        output = Activation('sigmoid')(combined)
        
        return Model(inputs=inputs, outputs=output)

 
    def _create_dcn(self, input_shape):
        """Create Deep & Cross Network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Cross network
        cross = inputs
        for _ in range(3):
            cross_product = Lambda(lambda x: 
                K.reshape(K.batch_dot(K.reshape(x[0], (-1, 1, K.shape(x[0])[1])),
                                    K.reshape(x[1], (-1, K.shape(x[1])[1], 1))),
                        (-1, 1)) * x[2])([inputs, cross, inputs])
            cross = Add()([cross, cross_product])
        
        # Deep network
        deep = Dense(256, activation='relu')(inputs)
        deep = BatchNormalization()(deep)
        deep = Dense(128, activation='relu')(deep)
        
        # Combine networks
        combined = Concatenate()([cross, deep])
        output = Dense(1, activation='sigmoid')(combined)
        
        return Model(inputs=inputs, outputs=output)

    def _create_catnet(self, input_shape):
        """Create CatNet for categorical data"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Feature embedding
        x = Dense(64, activation='relu')(inputs)
        x = BatchNormalization()(x)
        
        # Categorical processing blocks
        for units in [128, 256, 128]:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
        
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_deepgbm(self, input_shape):
        """Create DeepGBM hybrid model"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Deep component
        deep = Dense(256, activation='relu')(inputs)
        deep = BatchNormalization()(deep)
        deep = Dense(128, activation='relu')(deep)
        
        # GBM-like component
        gbm = Dense(64, activation='relu')(inputs)
        for _ in range(3):  # Multiple boosting stages
            residual = Dense(64, activation='relu')(gbm)
            gbm = Add()([gbm, residual])
        
        # Combine components
        combined = Concatenate()([deep, gbm])
        x = Dense(256, activation='relu')(combined)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_neural_decision_forest(self, input_shape):
        """Create Neural Decision Forest"""


        inputs = Input(shape=input_shape, name="input_main")
        
        # Feature transformation
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        
        # Decision paths
        trees = []
        for _ in range(5):  # Number of trees
            # Decision nodes
            decision = Dense(64, activation='sigmoid')(x)
            leaf = Dense(32, activation='relu')(decision)
            trees.append(leaf)
        
        # Combine trees by averaging using Lambda layer
        x = layers.Lambda(lambda x: tf.reduce_mean(x, axis=0))(trees)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)


    def _create_net_dnf(self, input_shape):
        """Create NetDNF model"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Feature transformation
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        
        # Decision paths (conjunctions)
        conjunctions = []
        for _ in range(5):  # Number of conjunctions
            # Conjunctions (decision nodes)
            decision = Dense(64, activation='sigmoid')(x)
            conjunction = Dense(32, activation='relu')(decision)
            conjunctions.append(conjunction)
        
        # Combine conjunctions by taking the maximum using Maximum layer
        x = layers.Maximum()(conjunctions)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)


    def compile_model(self, model):
        """Compile the model with appropriate optimizer and metrics"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name='auc')]
        )
        return model


    def _create_graph_attention_network(self, input_shape):
        """Create a Graph Attention Network (GAT)"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Create pseudo-adjacency using attention
        attention_scores = Dense(input_shape[0])(inputs)
        attention_scores = Softmax()(attention_scores)
        
        # Graph convolution operation
        x = inputs
        for units in [64, 128, 64]:
            # Multi-head attention for graph convolution
            heads = []
            for _ in range(4):  # Number of attention heads
                head = Dense(units)(x)
                head = Multiply()([head, attention_scores])
                heads.append(head)
            
            x = Concatenate()(heads)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

 
    def _create_graph_convolutional_network(self, input_shape):
        """Create a Graph Convolutional Network (GCN)"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Create adjacency matrix using feature similarity
        adj = Dense(input_shape[0])(inputs)
        adj = Activation('sigmoid')(adj)
        
        # Graph convolution layers
        x = inputs
        for units in [64, 128, 256]:
            # GCN operation: AXW
            x = Multiply()([adj, x])
            x = Dense(units)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_capsule_network(self, input_shape):
        """Create a Capsule Network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Primary capsules
        primary_caps = Dense(256)(inputs)
        primary_caps = Reshape((-1, 8))(primary_caps)  # 8D capsules
        
        # Squash activation for capsules
        def squash(vectors):
            s_squared_norm = K.sum(K.square(vectors), -1, keepdims=True)
            scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
            return scale * vectors
        
        primary_caps = Lambda(squash)(primary_caps)
        
        # Digit capsules
        digit_caps = Dense(16)(primary_caps)  # 16D capsules
        digit_caps = Lambda(squash)(digit_caps)
        
        # Length of the capsule outputs
        out_caps = Lambda(lambda x: K.sqrt(K.sum(K.square(x), -1)))(digit_caps)
        output = Dense(1, activation='sigmoid')(out_caps)
        
        return Model(inputs=inputs, outputs=output)

    def _create_dense_autoencoder(self, input_shape):
        """Create a Dense Autoencoder"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Encoder
        x = Dense(256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        
        # Latent space
        latent = Dense(32, activation='relu')(x)
        
        # Classification from latent space
        x = Dense(64, activation='relu')(latent)
        x = Dense(32, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)





    def _create_neural_turing_machine(self, input_shape):
        """Create a simplified Neural Turing Machine"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Controller (LSTM)
        controller = LSTM(128, return_sequences=True)(inputs)
        
        # Memory operations
        memory_size = 128
        memory = Dense(memory_size)(controller)
        
        # Read/write heads
        read_head = Dense(memory_size, activation='softmax')(controller)
        write_head = Dense(memory_size, activation='softmax')(controller)
        
        # Memory read/write operations
        read_content = Multiply()([memory, read_head])
        write_content = Multiply()([memory, write_head])
        
        memory_output = Add()([read_content, write_content])
        x = GlobalAveragePooling1D()(memory_output)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_highway_network(self, input_shape):
        """Create a Highway Network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Initial transformation to ensure consistent dimensions
        x = Dense(256)(inputs)
        
        for _ in range(5):  # Number of highway layers
            h = Dense(256, activation='relu')(x)
            t = Dense(256, activation='sigmoid')(x)
            c = Lambda(lambda x: 1.0 - x)(t)
            x = Add()([Multiply()([h, t]), Multiply()([x, c])])
            x = BatchNormalization()(x)
        
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)



    def _create_densenet(self, input_shape):
        """Create a DenseNet-style architecture"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Initial convolution
        x = Dense(64)(inputs)
        
        # Dense blocks
        for block in range(3):
            block_outputs = [x]
            for layer in range(4):  # Layers per block
                # Composite function
                h = BatchNormalization()(x)
                h = Activation('relu')(h)
                h = Dense(32)(h)
                
                # Concatenate with all previous outputs
                block_outputs.append(h)
                x = Concatenate()(block_outputs)
            
            # Transition layer
            if block < 2:  # No transition after last block
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Dense(x.shape[-1] // 2)(x)  # Compression
        
        x = GlobalAveragePooling1D()(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_deep_belief_network(self, input_shape):
        """Create a Deep Belief Network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Pre-training layers (RBM-like)
        x = inputs
        hidden_sizes = [256, 128, 64]
        
        for size in hidden_sizes:
            # Visible to hidden
            h = Dense(size, activation='sigmoid')(x)
            # Hidden to visible reconstruction
            v = Dense(x.shape[-1], activation='sigmoid')(h)
            # Fine-tuning
            x = Dense(size, activation='relu')(x)
            x = BatchNormalization()(x)
        
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_nested_lstm(self, input_shape):
        """Create a Nested LSTM architecture"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Initial Dense layer to match dimensions
        x = Dense(128)(inputs)
        x = Reshape((1, 128))(x)
        
        # Outer LSTM
        outer_lstm = LSTM(64, return_sequences=True)(x)
        
        # Simple Dense layer instead of nested LSTM
        x = Dense(32)(outer_lstm)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=output)

    def _create_temporal_convolutional_net(self, input_shape):
        """Create a Temporal Convolutional Network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        x = inputs
        n_filters = 64
        
        for dilation_rate in [1, 2, 4, 8]:
            residual = x
            # Dilated causal convolution
            x = Conv1D(n_filters, 3, padding='same', dilation_rate=dilation_rate)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.2)(x)
            
            # Residual connection
            if residual.shape[-1] != n_filters:
                residual = Conv1D(n_filters, 1, padding='same')(residual)
            x = Add()([x, residual])
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        return Model(inputs=inputs, outputs=output)

    def _create_neural_architecture_search(self, input_shape):
        """Create a simplified Neural Architecture Search Network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Create multiple parallel paths
        paths = []
        operations = [
            lambda x: Dense(64, activation='relu')(x),
            lambda x: Dense(128, activation='tanh')(x),
            lambda x: Dense(32, activation='selu')(x),
            lambda x: Dense(96, activation='elu')(x)
        ]
        
        for op in operations:
            path = op(inputs)
            path = BatchNormalization()(path)
            paths.append(path)
        
        # Combine paths
        x = Concatenate()(paths)
        
        # Apply Dense for attention (with softmax across paths)
        attention = Dense(len(paths), activation='softmax')(inputs)
        attention = Reshape((len(paths), 1))(attention)
        
        # Weighted sum of paths
        x = Reshape((len(paths), -1))(x)  # Reshape paths to match attention
        x = Multiply()([x, attention])  # Apply attention
        x = Reshape((-1,))(x)  # Flatten combined paths
        
        # Final Dense layers
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)  # Binary classification
        
        return Model(inputs=inputs, outputs=output)

    def _simple_build_fnn_model(self, input_shape):
        """Create a simple feedforward neural network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Dense layers with dropout
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
        return model

    def _simple_build_cnn_model(self, input_shape):
        """Create a simple convolutional neural network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Convolutional layers
        x = Reshape((input_shape[0], 1))(inputs)  # Add channel dimension
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
        return model

    def _simple_build_rnn_model(self, input_shape):
        print(f"Input shape: {input_shape}")  # Add this line
        inputs = Input(shape=input_shape, name="input_main")
        #x = Reshape((1, input_shape[0]))(inputs)  # Reshape for RNN
        x = SimpleRNN(64, activation='relu')(inputs)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
        return model

    def _simple_build_lstm_model(self, input_shape):
        """Create a simple LSTM network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # LSTM layer
        #x = Reshape(( 1,input_shape[0]))(inputs)  # Reshape for LSTM
        x = LSTM(64, activation='relu')(inputs)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
        return model

    def _simple_build_gru_model(self, input_shape):
        """Create a simple GRU network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # GRU layer
        #x = Reshape((1, input_shape[0]))(inputs)  # Reshape for GRU
        x = GRU(64, activation='relu')(inputs)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
        return model

    def _simple_build_bidirectional_lstm_model(self, input_shape):
        """Create a simple bidirectional LSTM network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Bidirectional LSTM layer
        #x = Reshape(( 1,input_shape[0]))(inputs)  # Reshape for LSTM
        x = Bidirectional(LSTM(64, activation='relu'))(inputs)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
        return model

    def _simple_build_1d_cnn_model(self, input_shape):
        """Create a simple 1D convolutional neural network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # 1D Convolutional layers
        x = Reshape((input_shape[0], 1))(inputs)  # Add channel dimension
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
        return model

    def _simple_build_mlp_model(self, input_shape):
        """Create a simple multilayer perceptron"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # MLP layers
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
        return model

    def _simple_build_resnet_model(self, input_shape):
        """Create a simple residual network"""
        inputs = Input(shape=input_shape, name="input_main")
        
        # Initial transformation
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        
        # Residual connection
        residual = Dense(64, activation='relu')(inputs)
        x = Add()([x, residual])
        
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
        return model

    



    def get_callbacks(self, model_name, monitor='val_auc'):
        """Get callbacks for training"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=10,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=5,
                mode='max'
            )
        ]


class DeepLearningEvaluator:
    def __init__(self, save_dir="model_results"):
        """Initialize evaluator"""
        self.available_models = [
            'DeepResDenseNet',
            'MultiScaleCNN',
            'HybridTransformerLSTM',
            'DenseTransformer',
            'DeepBiGRU',
            'PyramidNet',
            'MultiPathResNet',
            'AttentionLSTM',
            'DilatedCNN',
            'HierarchicalAttentionNet',
            'CrossStitchNet',
            'GatedRecurrentMixer',
            'MultiModalFusion',
            'DeepChannelNet',
            'SqueezeExciteNet',
            'DualPathNetwork',
            'CascadedNet',
            'DeepInteractionNet',
            'HybridResidualAttention',
            'MultiScaleFeatureFusion',
            'TabNet',
            'DeepFM',
         
            'NODE',
            'EntityEmbeddings',
            'FTTransformer',
            'WideAndDeep',
            
            'DCN',
            'CatNet',
            'DeepGBM',
            'NeuralDecisionForest',
            'NetDNF',
            'GraphAttentionNetwork',
            
            'GraphConvolutionalNetwork',
            'CapsuleNetwork',
            'DenseAutoencoder',
            'NeuralTuringMachine',
            'HighwayNetwork',
            'DenseNet',
            'DeepBeliefNetwork',
            'NestedLSTM',
            'TemporalConvolutionalNet',
            'NeuralArchitectureSearch',
            'simpleBuildFnnModel',
            'simpleBuildCnnModel',
            'simpleBuildRnnModel',
            'simpleBuildLstmModel',
            'simpleBuildGruModel',
            'simpleBuildBidirectionalLstmModel',
            'simpleBuild1dCnnModel',
            'simpleBuildMlpModel',
            'simpleBuildResnetModel'
        ]
        
        # Setup directories
        self.save_dir = save_dir
        self.plots_dir = f"{save_dir}/plots"
        self.models_dir = f"{save_dir}/models"
        self.logs_dir = f"{save_dir}/logs"
        
        # Create necessary directories
        for directory in [self.plots_dir, self.models_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

    def _reshape_data_for_model(self, model_name, model_factory, X_data):
            """Reshape input data based on model requirements"""
            if model_name in model_factory.models_needing_cnn_format:
                # Reshape for CNN models: (Samples, Features, 1)
                reshaped_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
                print(f"Reshaped data for CNN from {X_data.shape} to {reshaped_data.shape}")
                return reshaped_data
                
            elif model_name in model_factory.models_needing_3d:
                # Reshape for LSTM/Transformer: (Samples, 1, Features)
                reshaped_data = X_data.reshape(X_data.shape[0], 1, X_data.shape[1])
                print(f"Reshaped data for LSTM from {X_data.shape} to {reshaped_data.shape}")
                return reshaped_data
                
            return X_data

    def train_and_evaluate(self, X_train, X_test, X_val, y_train, y_test, y_val, 
                          epochs=100, batch_size=32, models_to_train=None):
        """Train and evaluate multiple deep learning models"""
        # Get input shape from the data
        input_shape = (X_train.shape[1],)  # Basic shape (Features,)
        
        # Initialize model factory
        model_factory = DeepLearningModels(input_shapes={'main': input_shape})
        
        results = []
        models_to_train = models_to_train or self.available_models
        
        for model_name in models_to_train:
            try:
                print(f"\nAttempting to train {model_name}...")
                training_start_time = datetime.now()
                
                # Create and compile model
                model = model_factory.create_model(model_name)
                model = model_factory.compile_model(model)
                
                # Reshape input data if needed
                X_train_model = self._reshape_data_for_model(model_name, model_factory, X_train)
                X_val_model = self._reshape_data_for_model(model_name, model_factory, X_val)
                X_test_model = self._reshape_data_for_model(model_name, model_factory, X_test)
                
                # Set up metrics callback
                metrics_callback = MetricsCallback(
                    validation_data=(X_val_model, y_val),
                    test_data=(X_test_model, y_test)
                )
                
                # Store training data for callback access
                model.train_data = (X_train_model, y_train)
                
                # Calculate class weights
                class_weights = self._calculate_class_weights(y_train)
                
                # Train model
                history = model.fit(
                    X_train_model,
                    y_train,
                    validation_data=(X_val_model, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[metrics_callback] + self._create_callbacks(model_name),
                    class_weight=class_weights,
                    verbose=0
                )
                
                # Calculate training time and metrics
                training_time = (datetime.now() - training_start_time).total_seconds()
                
                train_metrics = self._calculate_metrics(model, X_train_model, y_train)
                val_metrics = self._calculate_metrics(model, X_val_model, y_val)
                test_metrics = self._calculate_metrics(model, X_test_model, y_test)
                
                results.append({
                    'Model': model_name,
                    'Train AUC': train_metrics['auc'],
                    'Validation AUC': val_metrics['auc'],
                    'Test AUC': test_metrics['auc'],
                    'Training Time': training_time,
                    'Status': 'Success'
                })
                print(f"Successfully trained {model_name}")
                print(results[-1])
                
            except Exception as e:
                print(f"\nError training {model_name}:")
                print(f"Error details: {str(e)}")
                self._log_error(model_name, e)
                
                # Add failed model to results with null metrics
                results.append({
                    'Model': model_name,
                    'Train AUC': None,
                    'Validation AUC': None,
                    'Test AUC': None,
                    'Training Time': None,
                    'Status': 'Failed'
                })
                
                # Clean up memory for failed model
                import gc
                import tensorflow.keras.backend as K
                K.clear_session()
                gc.collect()
                
                # Continue with next model
                continue
                
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save final results
        results_df.to_csv(f"{self.logs_dir}/training_results.csv", index=False)
        return results_df

    def _calculate_metrics(self, model, X_data, y_true):
        """Calculate comprehensive metrics for model evaluation"""
        predictions = model.predict(X_data,verbose=0)
        return {
            'auc': roc_auc_score(y_true, predictions)
        }

    def _calculate_class_weights(self, y):
        """Calculate balanced class weights"""
        classes = np.unique(y)
        class_counts = np.bincount(y.astype(int))
        total_samples = len(y)
        weights = {i: total_samples / (len(classes) * count) 
                  for i, count in enumerate(class_counts)}
        return weights

    def _create_callbacks(self, model_name):
        """Create training callbacks"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=10,
                restore_best_weights=True,
                verbose=0 
                
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=5,
                mode='max',
                verbose=0   
            ) 
        ]

class MetricsCallback(tf.keras.callbacks.Callback):
    """Custom callback for tracking metrics during training"""
    def __init__(self, validation_data, test_data):
        super().__init__()
        self.validation_data = validation_data
        self.test_data = test_data
        self.history = {
            'train_auc': [], 'val_auc': [], 'test_auc': [],
            'train_loss': [], 'val_loss': [], 'test_loss': []
        }

    def on_epoch_end(self, epoch, logs={}):
        # Calculate metrics for all datasets
        val_pred = self.model.predict(self.validation_data[0],verbose=0)
        test_pred = self.model.predict(self.test_data[0],verbose=0)
        train_pred = self.model.predict(self.model.train_data[0],verbose=0)

        # Calculate AUC scores
        self.history['train_auc'].append(
            roc_auc_score(self.model.train_data[1], train_pred))
        self.history['val_auc'].append(
            roc_auc_score(self.validation_data[1], val_pred))
        self.history['test_auc'].append(
            roc_auc_score(self.test_data[1], test_pred))

        # Store losses
        self.history['train_loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        
        # Calculate test loss
        test_loss = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        self.history['test_loss'].append(test_loss[0])


def train_and_evaluate_deep_learning(X_train, X_test, X_val, y_train, y_test, y_val, 
                     epochs=100, batch_size=32, models_to_train=None):
    """Train and evaluate multiple deep learning models with per-model error handling"""
    evaluator = DeepLearningEvaluator(save_dir="experiment_results")
    results = []
    
    # Get list of models to train
    models_to_train = models_to_train or evaluator.available_models
    
    # Initialize model factory
    input_shape = (X_train.shape[1],)
    model_factory = DeepLearningModels(input_shapes={'main': input_shape})
    
    # Train each model independently
    for model_name in models_to_train:
        print(f"\nAttempting to train {model_name}...")
        try:
            # Create and compile model
            model = model_factory.create_model(model_name)
            model = model_factory.compile_model(model)
            
            # Reshape data if needed for this specific model
            try:
                X_train_model = evaluator._reshape_data_for_model(model_name, model_factory, X_train)
                X_val_model = evaluator._reshape_data_for_model(model_name, model_factory, X_val)
                X_test_model = evaluator._reshape_data_for_model(model_name, model_factory, X_test)
            except Exception as e:
                print(f"Error reshaping data for {model_name}: {str(e)}")
                # Add failed model to results with error status
                results.append({
                    'Model': model_name,
                    'Train AUC': None,
                    'Validation AUC': None,
                    'Test AUC': None,
                    'Training Time': None,
                    'Status': 'Failed - Data Reshaping Error',
                    'Error': str(e)
                })
                continue

            # Set up metrics callback
            metrics_callback = MetricsCallback(
                validation_data=(X_val_model, y_val),
                test_data=(X_test_model, y_test)
            )
            
            # Store training data for callback access
            model.train_data = (X_train_model, y_train)
            
            # Calculate class weights
            class_weights = evaluator._calculate_class_weights(y_train)
            
            # Start timing
            training_start_time = datetime.now()
            
            try:
                # Train model with error handling
                history = model.fit(
                    X_train_model,
                    y_train,
                    validation_data=(X_val_model, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[metrics_callback] + evaluator._create_callbacks(model_name),
                    class_weight=class_weights,
                    verbose=0,
                )
                
                # Calculate training time
                training_time = (datetime.now() - training_start_time).total_seconds()
                
                # Calculate final metrics
                train_metrics = evaluator._calculate_metrics(model, X_train_model, y_train)
                val_metrics = evaluator._calculate_metrics(model, X_val_model, y_val)
                test_metrics = evaluator._calculate_metrics(model, X_test_model, y_test)
                
                # Store successful results
                results.append({
                    'Model': model_name,
                    'Train AUC': train_metrics['auc'],
                    'Validation AUC': val_metrics['auc'],
                    'Test AUC': test_metrics['auc'],
                    'Training Time': training_time,
                    'Status': 'Success',
                    'Error': None
                })
                print(f"Successfully trained {model_name}")
                print(results[-1])
                
            except Exception as e:
                print(f"Error during training {model_name}: {str(e)}")
                # Add failed model to results with error status
                results.append({
                    'Model': model_name,
                    'Train AUC': None,
                    'Validation AUC': None,
                    'Test AUC': None,
                    'Training Time': None,
                    'Status': 'Failed - Training Error',
                    'Error': str(e)
                })
            
        except Exception as e:
            print(f"Error initializing {model_name}: {str(e)}")
            # Add failed model to results with error status
            results.append({
                'Model': model_name,
                'Train AUC': None,
                'Validation AUC': None,
                'Test AUC': None,
                'Training Time': None,
                'Status': 'Failed - Initialization Error',
                'Error': str(e)
            })
        
        finally:
            # Clean up memory after each model, regardless of success or failure
            try:
                import gc
                import tensorflow.keras.backend as K
                K.clear_session()
                gc.collect()
            except Exception as e:
                print(f"Warning: Error during cleanup after {model_name}: {str(e)}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save detailed results including errors
    try:
        results_df.to_csv(f"{evaluator.save_dir}/model_results_with_errors.csv", index=False)
    except Exception as e:
        print(f"Warning: Could not save results to CSV: {str(e)}")
    
    return results_df

# #def train_and_evaluate(X_train, X_test, X_val, y_train, y_test, y_val,epochs,batch_size):
# #    evaluator = DeepLearningEvaluator(save_dir="experiment_results")
 
# #    results = evaluator.train_and_evaluate(
#         X_train, X_test, X_val, y_train, y_test, y_val,
#         epochs,
#         batch_size,
#         )
#     import gc
#     import tensorflow.keras.backend as K

#     K.clear_session()
#     gc.collect()
#     return results

 
 