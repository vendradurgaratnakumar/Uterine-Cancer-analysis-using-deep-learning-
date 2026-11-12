import json
import pandas as pd
import os


json_file = "Dataset/clinical.project-tcga-ucec.2026-06-01.json"

# Load JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

patient_ids = []
grades = []

for case in data:
    patient_id = case.get("submitter_id")

    grade = None
    diagnoses = case.get("diagnoses", [])

    for diag in diagnoses:
        if diag.get("diagnosis_is_primary_disease") == "true":
            grade = diag.get("tumor_grade")
            break

    patient_ids.append(patient_id)
    grades.append(grade)


df = pd.DataFrame({
    "Patient_ID": patient_ids,
    "Grade": grades
})


dataset_path = "Dataset/TCGA-UCEC"

available_patients = {
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
}



df_filtered = df[df["Patient_ID"].isin(available_patients)].reset_index(drop=True)
# df_filtered.to_csv("TCGA_UCEC_Grades.csv", index=False)
















