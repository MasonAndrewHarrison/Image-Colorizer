import kaggle
import numpy as np 
import os
import shutil

"""
This uses the CIELAB (Lab) format 
and is from the MIRFLICKR-25K dataset.
"""
kaggle.api.authenticate()
kaggle.api.dataset_download_files("shravankumar9892/image-colorization", path="temp/all/", unzip=True)

test_size = 1000

L = (np.load("temp/all/l/gray_scale.npy") / 256) * 100

ab1 = np.load("temp/all/ab/ab/ab1.npy")
ab2 = np.load("temp/all/ab/ab/ab2.npy")
ab3 = np.load("temp/all/ab/ab/ab3.npy")

ab = np.concat([ab1, ab2, ab3], axis=0)

os.makedirs("CIELAB-Dataset/train", exist_ok=True)
os.makedirs("CIELAB-Dataset/test", exist_ok=True)

L_test = L[:test_size, :, :]
L_train = L[test_size:, :, :]
ab_test = ab[:test_size, :, : ,:]
ab_train = ab[test_size:, :, : ,:]

np.save("CIELAB-Dataset/test/L.npy", L_test)
np.save("CIELAB-Dataset/test/ab.npy", ab_test)
np.save("CIELAB-Dataset/train/L.npy", L_train)
np.save("CIELAB-Dataset/train/ab.npy", ab_train)
shutil.rmtree("temp/")