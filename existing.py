"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                         EXISTING
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""



# """======================================================================
# ========         DAVN- Dual attention vision network            =========
# ======================================================================"""


import numpy as np
import tensorflow as tf

import pickle
from xgboost import XGBClassifier

from tensorflow.keras.layers import (Input, Dense, Dropout, Multiply, Concatenate, GlobalAveragePooling1D,
    GlobalMaxPooling1D, Reshape, Conv1D, Add, Softmax, LayerNormalization, MultiHeadAttention, 
    GRU, BatchNormalization, Activation)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder







def Channel_Attention(x, reduction=8):

    channels = x.shape[-1]
    
    avg = GlobalAveragePooling1D()(x)
    avg = Dense(channels//reduction, activation='relu')(avg)

    avg = Dense(channels, activation='sigmoid')(avg)
    avg = Reshape((1,channels))(avg)

    out = Multiply()([x,avg])
    
    return out



def Spatial_Attention(x):

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = Reshape((1,-1))(avg_pool)
    max_pool = Reshape((1,-1))(max_pool)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention = Conv1D(filters=1, kernel_size=1, activation="sigmoid")(concat)

    out = Multiply()([x, attention])
    
    return out



def Vision_Attention(x):

    q = Dense(256)(x)
    k = Dense(256)(x)
    v = Dense(256)(x)

    score = tf.matmul(q, k, transpose_b=True)
    scale = tf.math.sqrt(tf.cast(256, tf.float32))
    score = score / scale

    attn = Softmax(axis=-1)(score)
    out = tf.matmul(attn,v)

    out = Dense(x.shape[-1])(out)

    return Add()([x,out])





def DAVN(input_dim, classes):

    inputs = Input(shape=(input_dim,))

    x = Dense(512, activation="relu")(inputs)
    x = Reshape((16,32))(x)

    x = Channel_Attention(x)
    x = Spatial_Attention(x)
    x = Vision_Attention(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation="relu")(x)
    outputs = Dense(classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="DAVN_Dual_Attention_Vision_Network")

    return model




def train_DAVN(X_train, y_train):

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    num_classes = len(np.unique(y_train))
    
    y_train = to_categorical(y_train, num_classes=num_classes)
    
    model = DAVN(input_dim=512, classes=num_classes)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2,)
    
    return model



def predict_DAVN(X_test):

    model = load_model("Models/DAVN.h5")
    
    pred_prob = model.predict(X_test)
    y_pred = np.argmax(pred_prob, axis=1)
    
    return y_pred




# """======================================================================
# ========           SwinNet- Swintransformer network             =========
# ======================================================================"""



def Swin_Transformer_Block(x, heads=4, dim=128):

    shortcut = x
    x = LayerNormalization()(x)

    attention = MultiHeadAttention(num_heads=heads, key_dim=dim)(x,x)
    x = Add()([shortcut, attention])

    shortcut = x
    x = LayerNormalization()(x)

    x = Dense(dim*2, activation="gelu")(x)
    x = Dense(dim)(x)

    x = Add()([shortcut, x])

    return x




def Patch_Embedding(x):

    x = Dense(512, activation="relu")(x)
    x = Reshape((16,32))(x)
    return x




def SwinNet(input_dim, classes):

    inputs = Input(shape=(input_dim,))

    x = Patch_Embedding(inputs)
    x = Dense(128)(x)

    x = Swin_Transformer_Block(x, heads=4, dim=128)
    x = Swin_Transformer_Block(x, heads=4, dim=128)

    x = GlobalAveragePooling1D()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation="relu")(x)
    outputs = Dense(classes, activation="softmax")(x)
    model = Model(inputs, outputs, name="SwinNet_SwinTransformer")

    return model




def train_SwinNet(X_train,y_train):

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    num_classes = len(np.unique(y_train))

    y_train = to_categorical(y_train, num_classes=num_classes)
    model = SwinNet(input_dim=512, classes=num_classes)

    model.compile(optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2,)

    return model



def predict_SwinNet(X_test):

    model = load_model("Models/SwinNet.h5")
    
    pred_prob = model.predict(X_test)
    y_pred = np.argmax(pred_prob, axis=1)

    return y_pred






# """==================================================================================
# ===========         GCRN- Graph convolutional recurrent network        ==============
# =================================================================================="""




def Graph_Convolution(x, units):

    x1 = Dense(units, activation="relu")(x)
    x2 = Dense(units, activation="relu")(x1)

    out = Add()([x1, x2])

    return out




def Graph_Recurrent_Block(x):

    x = Graph_Convolution(x, 128)
    x = GRU(128, return_sequences=True)(x)

    return x




def GCRN(input_dim, classes):

    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation="relu")(inputs)
    
    x = Reshape((16,32))(x)
    x = Graph_Recurrent_Block(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation="relu")(x)

    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(classes, activation="softmax")(x)
    model = Model(inputs, outputs, name="GCRN_Graph_Convolutional_Recurrent_Network")

    return model




def train_GCRN(X_train, y_train):

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    num_classes = len(np.unique(y_train))

    y_train = to_categorical(y_train, num_classes=num_classes)
    
    model = GCRN(input_dim=512, classes=num_classes)

    model.compile(optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2,)

    return model




def predict_GCRN(X_test):

    model = load_model("Models/GCRN.h5")

    pred_prob = model.predict(X_test)
    y_pred = np.argmax(pred_prob, axis=1)

    return y_pred







# """==================================================================================
# =======         ML-DenseNet- Machine learning (XGBoost) with DenseNet     ===========
# =================================================================================="""





def Dense_Block(x, units):

    x1 = Dense(units)(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)


    out = Concatenate()(
        [
            x,
            x1
        ]
    )

    return out





def DenseNet_Feature(input_dim, classes):


    inputs = Input(
        shape=(input_dim,)
    )


    x = Dense(
        256,
        activation="relu"
    )(inputs)



    x = Dense_Block(
        x,
        128
    )


    x = Dense_Block(
        x,
        128
    )


    x = Dense_Block(
        x,
        64
    )


    features = Dense(
        256,
        activation="relu",
        name="DenseNet_Features"
    )(x)



    output = Dense(
        classes,
        activation="softmax"
    )(features)



    model = Model(
        inputs,
        output,
        name="ML_DenseNet"
    )


    feature_model = Model(
        inputs,
        features
    )


    return model, feature_model





def train_ML_DenseNet(X_train,y_train):


    encoder = LabelEncoder()


    y_encoded = encoder.fit_transform(
        y_train
    )


    num_classes = len(
        np.unique(y_encoded)
    )


    y_cat = to_categorical(
        y_encoded,
        num_classes
    )



    # DenseNet classifier

    dense_model, feature_model = DenseNet_Feature(
        512,
        num_classes
    )



    dense_model.compile(

        optimizer=Adam(
            learning_rate=0.0001
        ),

        loss="categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )



    dense_model.fit(

        X_train,

        y_cat,

        epochs=1,

        batch_size=32,

        validation_split=0.2,

        shuffle=True
    )




    # Extract DenseNet features

    X_features = feature_model.predict(
        X_train
    )



    print(
        "Dense Features:",
        X_features.shape
    )





    xgb_model = XGBClassifier(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="multi:softmax",

        num_class=num_classes,

        eval_metric="mlogloss"

    )



    xgb_model.fit(

        X_features,

        y_encoded

    )

    return feature_model, xgb_model, encoder








def predict_ML_DenseNet(X_test):


    feature_model = load_model(
        "Models/DenseNet_Feature.h5"
    )



    with open(
        "Models/XGBoost.pkl",
        "rb"
    ) as f:

        xgb_model = pickle.load(f)



    with open(
        "Models/LabelEncoder.pkl",
        "rb"
    ) as f:

        encoder = pickle.load(f)




    X_features = feature_model.predict(
        X_test
    )



    pred = xgb_model.predict(
        X_features
    )



    y_pred = encoder.inverse_transform(
        pred
    )


    return y_pred








