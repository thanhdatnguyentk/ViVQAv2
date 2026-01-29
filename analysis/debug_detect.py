import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import os
import glob
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
IMAGE_DIR = os.path.join(ROOT_DIR, 'data', 'ds102', 'images')

def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        print("Loading model...")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        model.to(device)
        print("Model loaded.")
        
        image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        if not image_paths:
            print("No images found.")
            return

        img_path = image_paths[0]
        print(f"Processing {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(img_tensor)[0]
            
        print("Prediction keys:", prediction.keys())
        print("Success on 1 image.")
        
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
