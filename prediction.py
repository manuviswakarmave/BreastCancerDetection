import torch
import numpy as np
from torchvision.transforms import transforms
import torchvision.models as models
from PIL import Image
from joblib import load

import segmentation
import segmentation_models_pytorch as smp

# Load the trained model
segmentation_model = smp.Unet('resnet34', encoder_weights=None, in_channels=1,
                 classes=1)  # Define the same architecture as during training
segmentation_model.load_state_dict(
    torch.load('segmentation_model.pth', map_location=torch.device('cpu')))  # Load the trained weights
segmentation_model.eval()  # Set the model to evaluation mode

# Load pre-trained ResNet model
resnet = models.resnet18(pretrained=True)
# Remove the last layer (classification layer)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
# Set model to evaluation mode
resnet.eval()

# Load trained Random Forest model
random_forest = load('random_forest_model1.pkl')

# Define transformation to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size of ResNet
    transforms.ToTensor(),  # Convert to tensor
])

def predict(image):
    # Load and preprocess the input image
    # image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Extract features from the input image using ResNet
    with torch.no_grad():
        features = resnet(image).squeeze().numpy()

    # Make predictions using the trained Random Forest classifier
    prediction = random_forest.predict([features])

    return prediction[0]


