import urllib.request
import zipfile
import os
import shutil

def download():
    url = "https://github.com/vncorenlp/VnCoreNLP/archive/master.zip"
    zip_path = "vncorenlp.zip"
    target_dir = "analysis/vncorenlp"
    temp_dir = "analysis/vncorenlp_temp"
    
    # Download
    print("Downloading VnCoreNLP...")
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract
    print("Extracting...")
    os.makedirs(temp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    # Move files
    print("Moving files to target directory...")
    source_dir = os.path.join(temp_dir, "VnCoreNLP-master")
    os.makedirs(target_dir, exist_ok=True)
    
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(target_dir, item)
        if os.path.exists(d):
            if os.path.isdir(d):
                shutil.rmtree(d)
            else:
                os.remove(d)
        shutil.move(s, target_dir)
        
    # Cleanup
    print("Cleaning up...")
    os.remove(zip_path)
    shutil.rmtree(temp_dir)
    print("Done!")

if __name__ == "__main__":
    download()
