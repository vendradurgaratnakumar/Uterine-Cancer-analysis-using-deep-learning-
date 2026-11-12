"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                        EXISTING FILTERS
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""


"""========================================================================
                Ablation: SDDGE Without Diffusion
========================================================================"""

import numpy as np
import pywt
from scipy.fft import fft2
from cryptography.fernet import Fernet
import time



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



def generate_key():
    return Fernet.generate_key()



# ================================
# W/O Diffusion Encryption
# ================================

def SDDGE_without_diffusion_encrypt(images):

    encrypted=[]
    key_details=[]
    encryption_times=[]

    for i,img in enumerate(images):

        print("Encrypting",i+1)
        start=time.time()
        
        x = img.astype(np.float32)/255.0

        noise = np.random.normal(
            0,
            0.01,
            x.shape
        )
        
        
        zt = x + noise
        zt = np.clip(
            zt,
            0,
            1
        )

        key=generate_key()
        cipher=Fernet(key)

        encrypted_data=cipher.encrypt(
            zt.astype(np.float32).tobytes()
        )

        encrypted.append(encrypted_data)

        key_details.append({

            "key":key,
            "shape":x.shape,

            "fingerprint":
            {
            "entropy":entropy_feature(zt),
            "fft":fft_fingerprint(zt),
            "wavelet":wavelet_feature(zt)
            }

        })

        end=time.time()

        encryption_times.append(
            (end-start)*1000
        )

    return np.array(encrypted,dtype=object), key_details, encryption_times


# ================================
# W/O Diffusion Decryption
# ================================

def SDDGE_without_diffusion_decrypt(encrypted,key_details):

    recovered=[]
    decryption_times=[]

    for i,data in enumerate(encrypted):

        start=time.time()
        info=key_details[i]

        cipher=Fernet(info["key"])
        decrypted=cipher.decrypt(data)

        x=np.frombuffer(
            decrypted,
            dtype=np.float32
        )

        x=x.reshape(info["shape"])
        x=np.clip(x,0,1)
        x=x*255

        recovered.append(
            x.astype(np.uint8)
        )

        end=time.time()

        decryption_times.append(
            (end-start)*1000
        )
        
    return np.array(recovered),decryption_times





from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error, mean_absolute_error

preprocessed_images = np.load("Features/preprocessed_images.npy")
encrypted_images, keys, enc_times = SDDGE_without_diffusion_encrypt(preprocessed_images)
decrypted_images, dec_times = SDDGE_without_diffusion_decrypt(encrypted_images,keys)

original = preprocessed_images.astype(np.float32)
decrypted = decrypted_images.astype(np.float32)

if original.max() > 1:
    original = original / 255.0

if decrypted.max() > 1:
    decrypted = decrypted / 255.0



def metrics(original, decrypted):    
    
    MSE = []
    MAE = []
    PSNR = []
    SSIM = []
    
    for i in range(len(original)):
    
        img1 = original[i]
        img2 = decrypted[i]
    
        mse = np.mean((img1 - img2)**2)
        mae = np.mean(np.abs(img1-img2))
    
        if mse == 0:
            p = 100
        else:
            p = psnr(img1, img2, data_range=1.0)
    
        if img1.ndim == 3:
            s = ssim(img1, img2, channel_axis=-1, data_range=1.0, win_size=7)
        else:
            s = ssim(img1, img2, data_range=1.0, win_size=7)
            
        MSE.append(mse)
        MAE.append(mae)
        PSNR.append(p)
        SSIM.append(s)
        
    return MSE, MAE, PSNR, SSIM
        
        
        
MSE1, MAE1, PSNR1, SSIM1 = metrics(original, decrypted)


print("\n=========== SDDGE Without Diffusion - Security Analysis ===========")

print("Average MSE: ",np.mean(MSE1))
print("Average MAE: ",np.mean(MAE1))
print("Average PSNR(dB): ",np.mean(PSNR1))
print("Average SSIM: ",np.mean(SSIM1))







import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from SDDGE import SDDGE_encrypt, SDDGE_decrypt


preprocessed_images = np.load("Features/preprocessed_images.npy")
encrypted_images, keys, enc_times = SDDGE_encrypt(preprocessed_images)
decrypted_images, dec_times = SDDGE_decrypt(encrypted_images,keys)

original = preprocessed_images.astype(np.float32)
decrypted = decrypted_images.astype(np.float32)

if original.max() > 1:
    original = original / 255.0

if decrypted.max() > 1:
    decrypted = decrypted / 255.0



MSE2, MAE2, PSNR2, SSIM2 = metrics(original, decrypted)


print("\n=========== SDDGE Security Analysis ===========")

print("Average MSE: ",np.mean(MSE2))
print("Average MAE: ",np.mean(MAE2))
print("Average PSNR(dB): ",np.mean(PSNR2))
print("Average SSIM: ",np.mean(SSIM2))





import numpy as np

MSE1, MAE1, PSNR1, SSIM1 = metrics(original, decrypted)   
MSE2, MAE2, PSNR2, SSIM2 = metrics(original, decrypted)  



# Store mean values
security_results = {

    "SDDGE_Without_Diffusion":
    {
        "MSE": np.mean(MSE1),
        "MAE": np.mean(MAE1),
        "PSNR": np.mean(PSNR1),
        "SSIM": np.mean(SSIM1)
    },


    "SDDGE":
    {
        "MSE": np.mean(MSE2),
        "MAE": np.mean(MAE2),
        "PSNR": np.mean(PSNR2),
        "SSIM": np.mean(SSIM2)
    }

}


# np.save("Features/security_results.npy",security_results)




















