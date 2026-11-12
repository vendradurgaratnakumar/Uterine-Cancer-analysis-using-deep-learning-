import os
import cv2
import pydicom
import numpy as np

"""======================================================================
========                    Read CT and PT Data                 =========
======================================================================"""

dataset_path = "Dataset/TCGA-UCEC"

patients = []
patient_slices = []
patient_modalities = []
selected_roots = []

count = 0

for patient_id in sorted(os.listdir(dataset_path)):

    patient_path = os.path.join(dataset_path, patient_id)

    if not os.path.isdir(patient_path):
        continue

    count += 1
    print(f"\nProcessing Patient {count}: {patient_id}")

    folder_found = False

    # Search all folders inside this patient
    for root, dirs, files in os.walk(patient_path):

        dcm_files = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith(".dcm")
        ]

        # Ignore folders with <=10 DICOM files
        if len(dcm_files) <= 10:
            continue

        print(f"\nChecking Folder: {root}")
        print(f"Total DICOM files: {len(dcm_files)}")

        ct_count = 0
        pt_count = 0

        # Read every DICOM file
        for dcm_file in sorted(dcm_files):

            try:
                ds = pydicom.dcmread(dcm_file)

                modality = ds.get("Modality", "")

                # Keep only CT and PT
                if modality not in ["CT", "PT"]:
                    continue

                img = ds.pixel_array.astype(np.float32)
                img = cv2.resize(img, (128, 128))

                patients.append(patient_id)
                patient_slices.append(img)
                patient_modalities.append(modality)

                if modality == "CT":
                    ct_count += 1
                elif modality == "PT":
                    pt_count += 1

            except Exception:
                pass

        # Store folder only if CT/PT images were found
        if ct_count > 0 or pt_count > 0:
            selected_roots.append(root)
            folder_found = True

            print(f"CT slices : {ct_count}")
            print(f"PT slices : {pt_count}")

    if not folder_found:
        print("No CT/PT images found for this patient.")

patient_slices = np.asarray(patient_slices)

print("\n==================== Summary ====================")
print("Total Patients Processed :", count)
print("Total Images             :", len(patient_slices))
print("Total CT Images          :", patient_modalities.count("CT"))
print("Total PT Images          :", patient_modalities.count("PT"))
print("Folders containing CT/PT :", len(selected_roots))

# Optional saves
# np.save("Features/images.npy", patient_slices)
# np.save("Features/modalities.npy", np.array(patient_modalities))
# np.save("Features/patients.npy", np.array(patients))
# np.save("Features/selected_roots.npy", np.array(selected_roots, dtype=object), allow_pickle=True)





