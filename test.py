"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                        DATASET (TEST)
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""
import os
import cv2
import pydicom
import numpy as np
import tkinter as tk
import tensorflow as tf
from tkinter import filedialog

selected_roots = np.load("Features/selected_roots.npy", allow_pickle=True)

patient_map = {}

for path in selected_roots:
    path = str(path)
    parts = path.split(os.sep)
    for p in parts:
        if p.startswith("TCGA-"):
            patient_id = p
            break
    patient_map[patient_id] = path

root = tk.Tk()
root.withdraw()

base_folder = "Dataset/TCGA-UCEC"

print("\n")
print("<=========  CHOOSE PATIENT DATA  =========>")

selected_patient_folder = filedialog.askdirectory(title="Select Patient Folder",
    initialdir=base_folder)

if selected_patient_folder == "":
    raise Exception("No folder selected")

patient_id = os.path.basename(selected_patient_folder)

print("Selected Patient:")
print(patient_id)


if patient_id not in patient_map:
    raise Exception("Patient not found in selected_roots.npy")

dicom_folder = patient_map[patient_id]

print("\nMapped DICOM Folder:")
print(dicom_folder)



images = []

dcm_files = [
    os.path.join(dicom_folder,f)
    for f in os.listdir(dicom_folder)
    if f.lower().endswith(".dcm")
]


dcm_files = sorted(dcm_files)
print("\nNumber of slices:",len(dcm_files))


for file in dcm_files:

    try:
        ds = pydicom.dcmread(file)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.resize(img, (128,128))
        images.append(img)

    except Exception as e:
        print("Skipped:",file)

images = np.asarray(images)


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots(figsize=(5,5))
img_plot = ax.imshow(images[0], cmap="gray")
ax.axis("off")

title = ax.set_title( f"Slice 1 / {len(images)}")

def update(frame):
    img_plot.set_data(images[frame])
    title.set_text(f"Slice {frame+1} / {len(images)}")
    return [img_plot, title]

ani = FuncAnimation(fig, update, frames=len(images), interval=200, repeat=True)
plt.show()







print("\nPreprocessing...")
from preprocessing import adaptive_mean_phase_filter

preprocessed_images = []

for i, img in enumerate(images):
    filtered = adaptive_mean_phase_filter(img, kernel_size=5, phase_weight=0.3)
    preprocessed_images.append(filtered)

preprocessed_images = np.array(preprocessed_images)


import matplotlib.pyplot as plt

# Plot first preprocessed image
plt.figure(figsize=(5,5))

plt.imshow(preprocessed_images[0], cmap="gray")
plt.title("First Preprocessed Image")
plt.axis("off")
plt.show()



print("\nExtracting Features...")
from feature_extraction_1 import extract_features
features_1 = extract_features(preprocessed_images)

print("\nGenerating Hashes...")
from hashing import generate_npsh_hashes
hashes = generate_npsh_hashes(features_1, keep_ratio=4, num_segments=64, secret_key=2025)

print("\nVerifying...")
from blockchain_verification import verify
retrieved_ipfs, retrieved_keys = verify(hashes)

print("\nDownloading data from IPFS...")
from blockchain_verification import download_encrypted_data_from_ipfs
retrive_data = download_encrypted_data_from_ipfs(retrieved_ipfs)


print("\nDecrypting data...")
import numpy as np
from SDDGE import SDDGE_decrypt
retrive_decrypted_images, _ = SDDGE_decrypt(retrive_data, retrieved_keys)

print("\nExtracting Features...")
from feature_extraction_2 import extract_features
features_data = extract_features(retrive_decrypted_images)



print("\nDisplaying Feature Importance using LIME Explainable AI...")
from Lime import lime_explainer, img_lime
retrive_decrypted_images_first = retrive_decrypted_images[0]
lime_explainer(retrive_decrypted_images_first)
img_lime(retrive_decrypted_images_first)


print("\nDisplaying Image Heatmap using GRAD_CAM...")
from grad_cam import gradcam_plot
gradcam_plot(retrive_decrypted_images_first)


print("\nTesting...")
from quantum_classification import predict
pred = predict(features_data)

grade_map = {
    "G1": 0,
    "G2": 1,
    "G3": 2,
    "High Grade": 3
}

inv_grade_map = {v: k for k, v in grade_map.items()}
pred_grade = [inv_grade_map[i] for i in pred]

print("The predicted class is: ", pred_grade[0])





"=======================================  CALCULATE 20% DATA  ==========================================="


print("\nCalculating Proposed and Existing (20% data)...\n")


from existing import predict_DAVN, predict_SwinNet, predict_GCRN, predict_ML_DenseNet
from quantum_classification import predict
import time
import numpy as np

X_test = np.load("Features/X_test.npy")

start_time1 = time.time()
predicted_classes1 = predict(X_test)
end_time1 = time.time()
Detection_Time1 = end_time1 - start_time1
avg_inference_time1 = (Detection_Time1 / len(X_test)) * 1000

start_time2 = time.time()
predicted_classes2 = predict_DAVN(X_test)
end_time2 = time.time()
Detection_Time2 = end_time2 - start_time2
avg_inference_time2 = (Detection_Time2 / len(X_test)) * 1000

start_time3 = time.time()
predicted_classes3 =  predict_SwinNet(X_test)
end_time3 = time.time()
Detection_Time3 = end_time3 - start_time3
avg_inference_time3 = (Detection_Time3 / len(X_test)) * 1000

start_time4 = time.time()
predicted_classes4 = predict_GCRN(X_test)
end_time4 = time.time()
Detection_Time4 = end_time4 - start_time4
avg_inference_time4 = (Detection_Time4 / len(X_test)) * 1000

start_time5 = time.time()
predicted_classes5 = predict_ML_DenseNet(X_test)
end_time5 = time.time()
Detection_Time5 = end_time5 - start_time5
avg_inference_time5 = (Detection_Time5 / len(X_test)) * 1000


Detection_Time = np.stack([Detection_Time1, Detection_Time2, Detection_Time3, Detection_Time4, Detection_Time5])
Inference_Time = np.stack([avg_inference_time1, avg_inference_time2, avg_inference_time3, avg_inference_time4, avg_inference_time5])






from tensorflow.keras.models import load_model
from performance import calculate_flops, calculate_ram_consumption
from quantum_classification import QuantumEntanglementLayer


model1 = load_model("Models/Classifier_Model.h5", custom_objects={
        "QuantumEntanglementLayer": QuantumEntanglementLayer})
total_flops1, gflops1 = calculate_flops(model1)
ram_mb1 = calculate_ram_consumption(model1, X_test)

model2 = load_model("Models/DAVN.h5")
total_flops2, gflops2 = calculate_flops(model2)
ram_mb2 = calculate_ram_consumption(model2, X_test)

model3 = load_model("Models/SwinNet.h5")
total_flops3, gflops3 = calculate_flops(model3)
ram_mb3 = calculate_ram_consumption(model3, X_test)

model4 = load_model("Models/GCRN.h5")
total_flops4, gflops4 = calculate_flops(model4)
ram_mb4 = calculate_ram_consumption(model4, X_test)

model5 = load_model("Models/DenseNet_Feature.h5")
total_flops5, gflops5 = calculate_flops(model5)
ram_mb5 = calculate_ram_consumption(model5, X_test)

Flops = np.stack([gflops1, gflops2, gflops3, gflops4, gflops5])
Ram_Consumption = np.stack([ram_mb1, ram_mb2, ram_mb3, ram_mb4, ram_mb5])




"==== [  Performance Calculation  ] ===="  

from performance import plot
import numpy as np

y_test = np.load("Features/y_test.npy")

plot(y_test, predicted_classes1, predicted_classes2, predicted_classes3, predicted_classes4, 
     predicted_classes5, Detection_Time, Inference_Time, Flops, Ram_Consumption)



















