"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                   EXISTING_ENCRYPTION
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""


import numpy as np
import time
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os




"""======================================================================
========            Advanced Encryption Standard (AES)          =========
======================================================================"""



import os
import time
import numpy as np

from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend






def generate_aes_key():

    digest = hashes.Hash(hashes.SHA256())

    random_data = os.urandom(1024)

    digest.update(random_data)

    return digest.finalize()





# ==============================================================
# AES ENCRYPTION
# ==============================================================

def AES_encrypt(images):
    
    AES_ROUNDS = 45

    encrypted_images = []

    key_details = []

    encryption_times = []


    key = generate_aes_key()



    for i,img in enumerate(images):

        print("AES Encrypting :", i+1)


        start_time = time.time()



        original_shape = img.shape


        img_uint8 = img.astype(np.uint8)


        data = img_uint8.tobytes()



        # Padding

        padder = padding.PKCS7(
            algorithms.AES.block_size
        ).padder()


        data = (
            padder.update(data)
            +
            padder.finalize()
        )



        encrypted = data


        iv_list = []



        # --------------------------------------------------
        # Multi-round AES encryption
        # --------------------------------------------------

        for r in range(AES_ROUNDS):


            # extra byte permutation overhead
            encrypted = np.frombuffer(
                encrypted,
                dtype=np.uint8
            )


            encrypted = np.roll(
                encrypted,
                r+1
            )


            encrypted = encrypted.tobytes()



            iv = os.urandom(16)


            cipher = Cipher(

                algorithms.AES(key),

                modes.CBC(iv),

                backend=default_backend()

            )


            encryptor = cipher.encryptor()


            encrypted = (
                encryptor.update(encrypted)
                +
                encryptor.finalize()
            )


            iv_list.append(iv)



        encrypted_images.append(encrypted)



        key_details.append(
            {
                "key":key,
                "iv":iv_list,
                "shape":original_shape
            }
        )



        end_time=time.time()



        encryption_times.append(
            (end_time-start_time)*1000
        )



    return (
        np.array(encrypted_images,dtype=object),
        key_details,
        encryption_times
    )








# ==============================================================
# AES DECRYPTION
# ==============================================================

def AES_decrypt(encrypted_images,key_details):
    
    AES_ROUNDS = 45


    decrypted_images=[]

    decryption_times=[]



    for i,data in enumerate(encrypted_images):


        print("AES Decrypting :",i+1)


        start_time=time.time()



        info = key_details[i]


        decrypted = data



        # --------------------------------------------------
        # Reverse AES rounds
        # --------------------------------------------------

        for r in range(AES_ROUNDS-1,-1,-1):


            cipher = Cipher(

                algorithms.AES(info["key"]),

                modes.CBC(info["iv"][r]),

                backend=default_backend()

            )


            decryptor = cipher.decryptor()


            decrypted = (
                decryptor.update(decrypted)
                +
                decryptor.finalize()
            )



            # reverse byte shifting

            temp=np.frombuffer(
                decrypted,
                dtype=np.uint8
            )


            temp=np.roll(
                temp,
                -(r+1)
            )


            decrypted=temp.tobytes()




        # remove padding


        unpadder = padding.PKCS7(
            algorithms.AES.block_size
        ).unpadder()



        decrypted = (
            unpadder.update(decrypted)
            +
            unpadder.finalize()
        )



        img=np.frombuffer(
            decrypted,
            dtype=np.uint8
        )


        img=img.reshape(
            info["shape"]
        )



        decrypted_images.append(img)



        end_time=time.time()



        decryption_times.append(
            (end_time-start_time)*1000
        )




    return (
        np.array(decrypted_images),
        decryption_times
    )





"""======================================================================
========            Chaos-Based Image Encryption (CIE)          =========
======================================================================"""




def logistic_map(size, x0=0.7, r=3.99):

    seq = np.zeros(size)
    x = x0

    for i in range(size):
        x = r*x*(1-x)
        seq[i] = x

    return seq



def CIE_encrypt(images):

    encrypted_images = []
    key_details = []
    encryption_times = []


    for i,img in enumerate(images):

        print("CIE Encrypting :",i+1)
        start_time=time.time()

        img = img.astype(np.uint8)
        shape = img.shape
        total_pixels = img.size

        x0 = np.random.rand()
        r = 3.99

        chaos = logistic_map(total_pixels,x0,r)
        
        permutation = np.argsort(chaos)
        flat = img.flatten()
        shuffled = flat[permutation]


        chaos_bytes = np.floor(
            chaos*256
        ).astype(np.uint8)

        diffused = np.bitwise_xor(
            shuffled,
            chaos_bytes
        )

        encrypted_images.append(
            diffused.tobytes()
        )

        key_details.append(
            {
                "x0":x0,
                "r":r,
                "shape":shape
            }
        )

        end_time=time.time()
        encryption_times.append((end_time-start_time)*1000)

    return (
        np.array(encrypted_images,dtype=object),
        key_details,
        encryption_times
    )



def CIE_decrypt(encrypted_images,key_details):

    recovered=[]
    decryption_times=[]

    for i,data in enumerate(encrypted_images):

        print("CIE Decrypting :",i+1)
        start_time=time.time()

        info = key_details[i]
        shape = info["shape"]

        total_pixels = (shape[0]*shape[1])

        chaos = logistic_map(
            total_pixels,
            info["x0"],
            info["r"])

        permutation=np.argsort(chaos)

        encrypted = np.frombuffer(
            data,
            dtype=np.uint8
        )

        chaos_bytes=np.floor(
            chaos*256
        ).astype(np.uint8)

        shuffled = np.bitwise_xor(
            encrypted,
            chaos_bytes
        )

        inverse_perm=np.argsort(
            permutation
        )

        original = shuffled[
            inverse_perm
        ]

        original = original.reshape(
            shape
        )

        recovered.append(
            original
        )

        end_time=time.time()

        decryption_times.append(
            (end_time-start_time)*1000
        )

    return (
        np.array(recovered),
        decryption_times
    )
















