import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D, Reshape, Embedding,
    MultiHeadAttention, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model



class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):

        super().__init__()

        self.att = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)


    def call(self, x):

        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)

        return self.layernorm2(out1 + ffn_output)




def HashFormerNet(input_shape=(128,128,1), hash_bits=64):

    inputs = Input(shape=input_shape)

    # CNN Spatial Feature Extraction
    x = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    cnn_features = MaxPooling2D()(x)

    # Patch Embedding
    patches = Conv2D(256, kernel_size=4, strides=4, padding='same')(cnn_features)

    H = patches.shape[1]
    W = patches.shape[2]

    x = Reshape((H*W,256))(patches)

    # Positional Encoding
    positions = tf.range(start=0, limit=H*W, delta=1)
    pos_embed = Embedding(input_dim=H*W, output_dim=256)(positions)

    x = x + pos_embed

    # Trans-ConVformer Blocks
    for _ in range(4):
        x = TransformerBlock(embed_dim=256, num_heads=8, ff_dim=512)(x)

    # Global Semantic Feature
    semantic_features = GlobalAveragePooling1D()(x)
    semantic_features = Dense(512, activation='gelu')(semantic_features)
    semantic_features = Dropout(0.3)(semantic_features)

    # NPSH Encoder
    z1 = Dense(256, activation='relu')(semantic_features)
    z2 = Dense(128, activation='relu')(z1)

    # Hash Layer
    hash_logits = Dense(hash_bits, activation='tanh', name='hash_logits')(z2)

    # Binary Hash Code
    hash_code = Lambda(lambda x: tf.sign(x), name='binary_hash')(hash_logits)
    model = Model(inputs, [semantic_features, hash_logits, hash_code])

    return model




def build_extractor():
    
    model = HashFormerNet(input_shape=(128,128,1), hash_bits=64)
    # model.summary()
    
    return model



def extract_features(X):
    
    model = load_model("Models/HashFormerNet_Extractor.h5",     
                       custom_objects={"TransformerBlock": TransformerBlock})
    
    X = np.expand_dims(X, axis=-1)
    deep_features, _, _ = model.predict(X, batch_size=1, verbose=1)
    
    return deep_features











