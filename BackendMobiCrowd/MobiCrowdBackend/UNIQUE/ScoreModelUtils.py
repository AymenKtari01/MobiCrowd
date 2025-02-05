import torch
from torchvision import transforms
from MobiCrowdBackend.UNIQUE.BaseCNN import BaseCNN
from MobiCrowdBackend.UNIQUE.Transformers import AdaptiveResize
from PIL import Image
import numpy as np

class Config:
    def __init__(self, backbone, fc=True, representation='BCNN', std_modeling=True):
        self.backbone = backbone
        self.fc = fc
        self.representation = representation
        self.std_modeling = std_modeling

# Use CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transformations
test_transform = transforms.Compose([
    AdaptiveResize(768),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

def load_score_model():
    ckpt_path = 'MobiCrowdBackend/UNIQUE/model.pt' 
    
    config = Config(backbone='resnet34', fc=True, representation='BCNN', std_modeling=True)
    model = BaseCNN(config)

    model = torch.nn.DataParallel(model).to(device)

    try:
        checkpoint = torch.load(ckpt_path , map_location=device)
        model.load_state_dict(checkpoint)
        print("Score Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        raise

    model.eval()
    return model

def evaluate(model, image_array):
    try:
        # Ensure the array is in the correct format
        image_array = np.array(image_array)
        if image_array.ndim != 3 or image_array.shape[2] not in [1, 3]:
            raise ValueError("Image array must be a 3D array with 1 or 3 channels.")

        image = Image.fromarray(image_array.astype('uint8'))
        image = test_transform(image)
        image = torch.unsqueeze(image, dim=0).to(device)
        
        with torch.no_grad():
            # Model output
            score, std = model(image)
            
        return score.cpu().item(), std.cpu().item()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
