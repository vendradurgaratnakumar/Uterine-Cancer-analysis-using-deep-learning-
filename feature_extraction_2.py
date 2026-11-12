"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                     Feature Pyramid Swin Transformer
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""



import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Add, LayerNormalization, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.layers import MultiHeadAttention



def patch_partition(x, patch=4):
    
    x=tf.image.extract_patches(images=x,sizes=[1,patch,patch,1],
                               strides=[1,patch,patch,1],rates=[1,1,1,1],padding="VALID")
    B=tf.shape(x)[0]
    
    return tf.reshape(x,[B,-1,x.shape[-1]])



def window_partition(x, w):

    N=tf.shape(x)[1]
    H=tf.cast(tf.sqrt(tf.cast(N,tf.float32)),tf.int32)
    C=x.shape[-1]
    x=tf.reshape(x,[-1,H,H,C])
    x=tf.image.extract_patches(images=x,sizes=[1,w,w,1],strides=[1,w,w,1],
                               rates=[1,1,1,1],padding="VALID")
    
    return tf.reshape(x,[-1,w*w,C])



def window_reverse(x,w,H):
    
    B=tf.shape(x)[0]//((H//w)**2)
    C=x.shape[-1]
    x=tf.reshape(x,[B,H//w,H//w,w,w,C])
    x=tf.transpose(x,[0,1,3,2,4,5])
    x=tf.reshape(x,[B,H,H,C])
    return tf.reshape(x,[B,H*H,C])



def swin_attention(x,dim,heads,w,shift=False):
    
    H=tf.cast(tf.sqrt(tf.cast(tf.shape(x)[1],tf.float32)),tf.int32)

    if shift:
        img=tf.reshape(x,[-1,H,H,dim])
        img=tf.roll(img,[-w//2,-w//2],[1,2])
        x=tf.reshape(img,[-1,H*H,dim])

    win=window_partition(x,w)
    attn=MultiHeadAttention(num_heads=heads,key_dim=dim//heads)
    win=attn(win,win)
    
    return window_reverse(win,w,H)



def swin_block(x,dim,heads=4,w=4,shift=False):
    
    skip=x
    x=LayerNormalization()(x)
    x=swin_attention(x,dim,heads,w,shift)
    x=Add()([skip,x])

    skip=x
    x=LayerNormalization()(x)
    x=Dense(dim*4,activation="gelu")(x)
    x=Dense(dim)(x)
    
    return Add()([skip,x])


def patch_merge(x,dim):
    
    N=tf.shape(x)[1]
    H=tf.cast(tf.sqrt(tf.cast(N,tf.float32)),tf.int32)
    C=x.shape[-1]

    x=tf.reshape(x,[-1,H,H,C])

    x=tf.concat([
        x[:,0::2,0::2,:],
        x[:,1::2,0::2,:],
        x[:,0::2,1::2,:],
        x[:,1::2,1::2,:]
    ],axis=-1)

    x=tf.reshape(x,[-1,(H//2)*(H//2),4*C])
    
    return Dense(dim*2)(x)


def FPSwinTransformer(input_shape=(128,128,1),C=32):

    inp=Input(input_shape)

    x=patch_partition(inp,4)
    x=Dense(C)(x)

    s1=swin_block(x,C,4,4)

    x=patch_merge(s1,C)
    s2=swin_block(x,C*2,4,4,True)

    x=patch_merge(s2,C*2)
    s3=swin_block(x,C*4,8,4)

    x=patch_merge(s3,C*4)
    s4=swin_block(x,C*8,8,4,True)

    f1=GlobalAveragePooling1D()(s1)
    f2=GlobalAveragePooling1D()(s2)
    f3=GlobalAveragePooling1D()(s3)
    f4=GlobalAveragePooling1D()(s4)

    x=Concatenate()([f1,f2,f3,f4])
    out=Dense(512,activation="relu")(x)

    return Model(inp,out)



def build_model():
    model = FPSwinTransformer()
    return model


def extract_features(X): 
    
    model = load_model("Models/FPSwinPAM_Extractor.h5",compile=False)
    X=X[...,None]
    
    features = model.predict(X, batch_size=1)
    
    return features










