"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                        DATASET (TRAIN)
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""

import os
import cv2
import pydicom
import numpy as np


"""======================================================================
========                         Read Data                      =========
======================================================================"""

dataset_path = "Dataset/TCGA-UCEC"

patients = []
patient_slices = []
selected_roots = []

count = 0

for patient_id in sorted(os.listdir(dataset_path)):

    patient_path = os.path.join(dataset_path, patient_id)

    if not os.path.isdir(patient_path):
        continue

    count += 1
    print(f"Processing Patient {count}: {patient_id}")

    folder_found = False

    # Search all series folders inside this patient
    for root, dirs, files in os.walk(patient_path):

        dcm_files = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith(".dcm")
        ]

        # Ignore series having <= 10 slices
        if len(dcm_files) <= 10:
            continue

        print(f"  Using Folder: {root}")
        print(f"  Number of slices: {len(dcm_files)}")
        
        selected_roots.append(root)

        # Sort DICOM files
        dcm_files = sorted(dcm_files)

        # Read ALL slices from ONLY THIS FOLDER
        for dcm_file in dcm_files:

            try:
                ds = pydicom.dcmread(dcm_file)

                img = ds.pixel_array.astype(np.float32)

                # Resize slice
                img = cv2.resize(img, (128, 128))

                patients.append(patient_id)
                patient_slices.append(img)

            except Exception:
                pass

        folder_found = True
        break   # <-- stop after first valid folder

    if not folder_found:
        print("No valid folder (>10 DICOM files) found")

patient_slices = np.asarray(patient_slices)
# np.save("Features/images.npy", patient_slices)

selected_roots = np.array(selected_roots, dtype=object)
# np.save("Features/selected_roots.npy", selected_roots, allow_pickle=True)





"""======================================================================
========                      label creation                    =========
======================================================================"""

import pandas as pd
import numpy as np

# Read CSV
df = pd.read_csv("Dataset/TCGA_UCEC_Grades.csv")

# Grade mapping
grade_map = {
    "G1": 0,
    "G2": 1,
    "G3": 2,
    "High Grade": 3
}

# Create Patient_ID -> Label dictionary
patient_label_dict = {
    pid: grade_map[grade]
    for pid, grade in zip(df["Patient_ID"], df["Grade"])
    if grade in grade_map
}

# Assign label for each collected slice
labels = []

for patient_id in patients:

    if patient_id in patient_label_dict:
        labels.append(patient_label_dict[patient_id])
    else:
        labels.append(-1)  



labels = np.array(labels)
# np.save("Features/labels.npy", labels)





"""======================================================================
====          Preprocessing (Adaptive Mean Phase Filter)           ======
======================================================================"""

import numpy as np
import cv2
from preprocessing import adaptive_mean_phase_filter


images = np.load("Features/images.npy")

preprocessed_images = []
count = 0

for i, img in enumerate(images):

    filtered = adaptive_mean_phase_filter(img, kernel_size=5, phase_weight=0.3)
    count = count+1
    print(count)
    preprocessed_images.append(filtered)

preprocessed_images = np.array(preprocessed_images)
# np.save("Features/preprocessed_images.npy", preprocessed_images)





"""===============================================================================
====          Feature Extraction (CNN-Transformer-based HashFormerNet)      ======
==============================================================================="""

from feature_extraction_1 import build_extractor, extract_features
import numpy as np

preprocessed_images = np.load("Features/preprocessed_images.npy")

model = build_extractor()
# model.save("Models/HashFormerNet_Extractor.h5")

deep_features = extract_features(preprocessed_images)
# np.save("Features/HashFormerNet_Features.npy", deep_features)





"""===============================================================================
====          Hash Generation (Neuro-Perceptual Secure Hashing (NPSH))      ======
==============================================================================="""

from hashing import generate_npsh_hashes
import numpy as np

deep_features = np.load("Features/HashFormerNet_Features.npy")
npsh_hashes = generate_npsh_hashes(deep_features, keep_ratio=4, num_segments=64, secret_key=2025)

# np.save("Features/npsh_hashes.npy",npsh_hashes)




"""===============================================================================
==   Image Encryption (Secure Diffusion Dynamic Generative Encryption (SD2GE))  ==
==============================================================================="""

import numpy as np
from SDDGE import SDDGE_encrypt

preprocessed_images = np.load("Features/preprocessed_images.npy")
encrypted_images, keys, enc_times = SDDGE_encrypt(preprocessed_images)

# np.save("Features/enc_times.npy", enc_times)
# np.save("Features/encrypted_images.npy", encrypted_images)
# np.save("Features/sddge_keys.npy", keys, allow_pickle=True)





"====  Encryption Existing  ===="

from encryption_existing import AES_encrypt, AES_decrypt

aes_encrypted, aes_keys, aes_enc_time = AES_encrypt(preprocessed_images)
aes_decrypted, aes_dec_time = AES_decrypt(aes_encrypted,aes_keys)

# np.save("Features/AES_enc_times.npy", aes_enc_time)
# np.save("Features/AES_dec_times.npy", aes_dec_time)

from encryption_existing import CIE_encrypt, CIE_decrypt

cie_encrypted, cie_keys, cie_enc_time = CIE_encrypt(preprocessed_images)
cie_decrypted, cie_dec_time = CIE_decrypt(cie_encrypted,cie_keys)

# np.save("Features/CIE_enc_times.npy", cie_enc_time)
# np.save("Features/CIE_dec_times.npy", cie_dec_time)




"""===============================================================================
============         Enhanced Ancestor Merkle Tree (EAMT)           ==============
==============================================================================="""

import numpy as np
import json
from merkle_tree import EAMT


hashes = np.load("Features/npsh_hashes.npy")
record, tree, root_hash = EAMT(hashes, stride=4)

# np.save("Features/EAMT_tree.npy", tree, allow_pickle=True)
# np.save("Features/root_hash.npy", root_hash, allow_pickle=True)



"""===============================================================================
============           Store Encrypted Images in IPFS               ==============
==============================================================================="""

import numpy as np 
import requests

encrypted_images = np.load("Features/encrypted_images.npy",allow_pickle=True)


def add_to_ipfs(binary):
    
    url = "http://127.0.0.1:5001/api/v0/add"
    response = requests.post(url, files={"file": binary})
    ipfs_hash = response.json()["Hash"]
    
    return ipfs_hash


def upload_encrypted_data_to_ipfs(encrypted_data):
    
    ipfs_hash_list = []
    for index, encrypted_img in enumerate(encrypted_data):
        
        try:
            # Upload the encrypted data to IPFS
            ipfs_hash = add_to_ipfs(encrypted_img)
            ipfs_hash_list.append(ipfs_hash)
            print(f"Uploaded {index + 1}/{len(encrypted_data)} → IPFS Hash: {ipfs_hash}")
            
        except Exception as e:
            print(f"Error uploading image {index}: {e}")
            ipfs_hash_list.append(None)
    
    return ipfs_hash_list

ipfs_hashes = upload_encrypted_data_to_ipfs(encrypted_images)

# np.save("Features/ipfs_hashes_new.npy", ipfs_hashes, allow_pickle=True)





"""===============================================================================
============              Store details in Blockchain               ==============
==============================================================================="""

import numpy as np
from datetime import datetime
import json


ipfs_hashes = np.load("Features/ipfs_hashes_new.npy")
npsh_hashes = np.load("Features/npsh_hashes.npy")
root_hash = np.load("Features/root_hash.npy")
keys = np.load("Features/sddge_keys.npy",allow_pickle=True)

images = np.load("Features/preprocessed_images.npy")
labels = np.load("Features/labels.npy")


# Create aligned records for each image
image_records = []

for i in range(len(images)):i
    record = {
        "image_index": i,
        "label": int(labels[i]) if i < len(labels) else None,
        "npsh_hash": npsh_hashes[i],
        "ipfs_hash": ipfs_hashes[i],
        "key": keys[i],   
        "timestamp": datetime.utcnow().isoformat()
    }
    
    image_records.append(record)


# Create blockchain record with Merkle root and all IPFS hashes
blockchain_record = {
    "MerkleRoot": root_hash,
    "TotalImages": len(images),
    "IPFS_Hashes": ipfs_hashes, 
    "NPSH_Hashes": npsh_hashes,  
    "Keys": keys,
    "Timestamp": datetime.utcnow().isoformat(),
    "ImageRecords": image_records 
}


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, np.generic):
        return obj.item()

    elif isinstance(obj, bytes):
        return obj.decode("utf-8")

    else:
        return obj


blockchain_record_json = json.dumps(blockchain_record, default=convert_numpy,
    indent=4)

# with open("Features/blockchain_record.json", "w") as f:
#     f.write(blockchain_record_json)




"""===============================================================================
========               Retrieve Encrypted Image From IPFS                =========
==============================================================================="""


import json

" Retrieve Image From IPFS "

def get_from_ipfs(ipfs_hash):
    
    url = f"http://127.0.0.1:5001/api/v0/cat?arg={ipfs_hash}"
    response = requests.post(url)
    
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to retrieve data: {response.text}")
 
    
def download_encrypted_data_from_ipfs(ipfs_hash_list):
    encrypted_data = []
    
    for index, ipfs_hash in enumerate(ipfs_hash_list):
        try:
            if ipfs_hash is not None:
                data = get_from_ipfs(ipfs_hash)
                encrypted_data.append(data)
                print(f"Downloaded {index + 1}/{len(ipfs_hash_list)}")
            else:
                print(f"Hash missing for image {index}")
                encrypted_data.append(None)
        except Exception as e:
            print(f"Error downloading {ipfs_hash}: {e}")
            encrypted_data.append(None)

    return encrypted_data



# with open("Features/blockchain_record.json", "r") as f:
#     blockchain_record = json.load(f)


# Download using the stored IPFS hashes from blockchain record
encrypted_images = download_encrypted_data_from_ipfs(blockchain_record['IPFS_Hashes'])
encrypted_images = np.asarray(encrypted_images)





"""===============================================================================
========                         Image Decryption                        =========
==============================================================================="""

import numpy as np
from SDDGE import SDDGE_decrypt


keys = np.load("Features/sddge_keys.npy",allow_pickle=True)
decrypted_images, dec_times = SDDGE_decrypt(encrypted_images, keys)

# np.save("Features/decrypted_images.npy", decrypted_images)
# np.save("Features/dec_times.npy", dec_times)




"""===============================================================================
========    Feature Extraction  (Feature Pyramid Swin Transformer)       =========
==============================================================================="""

from feature_extraction_2 import build_model, extract_features
import tensorflow as tf
import numpy as np

decrypted_images = np.load("Features/decrypted_images.npy")


model = build_model()
# model.save("Models/FPSwinPAM_Extractor.h5")

features = extract_features(decrypted_images)
# np.save("Features/FPSwinPAM_Features.npy", features)


"GRAD-CAM MODEL"

model = tf.keras.models.load_model("Models/FPSwinPAM_Extractor.h5")
target_layer = "layer_normalization_5"
grad_model = tf.keras.models.Model(
    inputs=model.input, outputs=[model.get_layer(target_layer).output, model.output])
# model.save("Models/FPSwinPAM_grad.npy")


"""===============================================================================
========               Train Test Split (80% Train, 20% Test)            =========
==============================================================================="""

import numpy as np
from sklearn.model_selection import train_test_split

# Load data
features = np.load("Features/FPSwinPAM_Features.npy")
labels = np.load("Features/labels.npy")

# 80% Train, 20% Test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# np.save("Features/X_train.npy", X_train)
# np.save("Features/X_test.npy", X_test)
# np.save("Features/y_train.npy", y_train)
# np.save("Features/y_test.npy", y_test)




"""===============================================================================
===     Classification (Hybrid Quantum Entanglement Deep Model (HQEDM))        ===
==============================================================================="""

import numpy as np
from quantum_classification import train_model

X_train = np.load("Features/X_train.npy")
y_train = np.load("Features/y_train.npy")

model = train_model(X_train, y_train)
# model.save("Models/Classifier_Model.h5")




"""===============================================================================
================                 Existing (Training)             =================
==============================================================================="""


from existing import train_DAVN, train_SwinNet, train_GCRN, train_ML_DenseNet
import numpy as np

X_train = np.load("Features/X_train.npy")
y_train = np.load("Features/y_train.npy")


model1 = train_DAVN(X_train, y_train)
# model1.save("Models/DAVN.h5")

model2 = train_SwinNet(X_train, y_train)
# model2.save("Models/SwinNet.h5")

model3 = train_GCRN(X_train, y_train)
# model3.save("Models/GCRN.h5")

feature_model, xgb_model, encoder = train_ML_DenseNet(X_train, y_train)

# feature_model.save("Models/DenseNet_Feature.h5")

# with open("Models/XGBoost.pkl","wb") as f:
#     pickle.dump(xgb_model,f)

# with open("Models/LabelEncoder.pkl","wb") as f:
#     pickle.dump(encoder,f)






















