import os
import json
import shutil
from tqdm import tqdm

DATA_DIR = "./data/ds102/images"

def check_leng_images(dir):
    files = os.listdir(dir)
    print(f"Total images: {len(files)}")

check_leng_images(DATA_DIR)
