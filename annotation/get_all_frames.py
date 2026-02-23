import os, glob, shutil

####################################################

STUDENT_NO = "25100812"
SURNAME = "Liang"

dataset_folder = f"dataset/{STUDENT_NO}_{SURNAME}"
annotating_folder = "dataset/labelling/images/"
label_folder = annotating_folder.replace("images", "labels")

####################################################
####################################################

os.makedirs(annotating_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

files = glob.glob(f"{dataset_folder}/*/*/rgb/*.png")

for file in files:
    new_file = annotating_folder+file.replace(dataset_folder, "")[1:].replace("\\", "_")
    print(f"copying {file} to {new_file}")
    shutil.copyfile(file, new_file)