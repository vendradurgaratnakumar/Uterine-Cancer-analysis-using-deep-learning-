"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                              Secure Diffusion Dynamic Generative Encryption
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""



import numpy as np
import pywt
from scipy.fft import fft2
from cryptography.fernet import Fernet
import time


# Diffusion Scheduler
def diffusion_scheduler(t, T):

    beta = np.linspace(0.0001, 0.005, T)
    alpha = np.sqrt(1-beta[t])
    sigma = np.sqrt(beta[t])

    return alpha, sigma


# Entropy
def entropy_feature(img):

    hist,_ = np.histogram(img, bins=256, range=(0,1), density=True)
    hist += 1e-12

    return -np.sum(hist*np.log(hist))


# FFT Fingerprint
def fft_fingerprint(img):

    f = fft2(img)
    mag=np.abs(f)

    return np.mean(mag), np.std(mag)


# Wavelet Fingerprint
def wavelet_feature(img):

    coeff = pywt.wavedec2(img,'haar',level=2)
    feat=[]

    for c in coeff:
        if isinstance(c,tuple):

            for x in c:
                feat.append(np.mean(np.abs(x)))

        else:
            feat.append(np.mean(np.abs(c)))

    return np.array(feat)


# Key generation
def generate_key():
    return Fernet.generate_key()


# SDDGE Encryption
def SDDGE_encrypt(images, diffusion_steps=50):

    encrypted = []
    key_details = []
    encryption_times = []

    for i,img in enumerate(images):
        
        print("Encrypting", i+1)
        
        start_time = time.time()

        # keep original float
        x = img.astype(np.float32)

        # normalize
        x = x / 255.0

        # choose low noise timestep
        t=np.random.randint(0,10)
        alpha, sigma = diffusion_scheduler(t,diffusion_steps)

        # fixed noise seed
        seed=np.random.randint(1, 999999999)
        rng=np.random.RandomState(seed)

        noise=rng.normal(0, 1, x.shape)

        # diffusion
        zt = alpha*x + sigma*noise

        # encryption
        key=generate_key()
        cipher=Fernet(key)

        encrypted_data=cipher.encrypt(zt.astype(np.float32).tobytes())
        encrypted.append(encrypted_data)

        key_details.append({
            "key":key,
            "alpha":alpha,
            "sigma":sigma,
            "seed":seed,
            "shape":x.shape,
            "fingerprint":
            {
            "entropy":entropy_feature(zt),
            "fft":fft_fingerprint(zt),
            "wavelet":wavelet_feature(zt)
            }
        })
            
        end_time = time.time()
        encryption_times.append((end_time - start_time) * 1000)

    encrypted = np.array(encrypted,dtype=object)

    return encrypted, key_details, encryption_times



# SDDGE Decryption
def SDDGE_decrypt(encrypted, key_details):

    recovered=[]
    decryption_times = []

    for i,data in enumerate(encrypted):
        
        start_time = time.time()

        info=key_details[i]
        cipher=Fernet(info["key"])

        # decrypt
        zt_bytes=cipher.decrypt(data)
        zt=np.frombuffer(zt_bytes, dtype=np.float32)

        zt=zt.reshape(info["shape"])

        # regenerate SAME noise
        rng=np.random.RandomState(info["seed"])

        noise=rng.normal(0, 1, zt.shape)

        # reverse diffusion
        x=(zt - info["sigma"]*noise) / info["alpha"]
        x=np.clip(x, 0, 1)

        x=x*255
        recovered.append(x.astype(np.uint8))
        
        end_time = time.time()
        decryption_times.append((end_time - start_time) * 1000)
        

    return np.array(recovered), decryption_times

