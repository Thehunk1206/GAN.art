import os
import shutil

base_dir = "dataset/"

if not os.path.exists("small_dataset/"):
    print("creating small_dataset")
    os.mkdir("small_dataset/")

img_files = os.listdir(base_dir)[:15000]

for img in img_files:
    shutil.copy(f"{base_dir}{img}","small_dataset/")
    print(f"{base_dir}{img} -->copies to small_dataset/{img} ")
print("done")
