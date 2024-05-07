import os

import torch
import torchvision.transforms as transforms
from PIL import Image
import segmentation_models_pytorch as smp


# Load the trained model
model = smp.Unet('resnet34', encoder_weights=None, in_channels=1,
                 classes=1)  # Define the same architecture as during training
model.load_state_dict(
    torch.load('segmentation_model.pth', map_location=torch.device('cpu')))  # Load the trained weights
model.eval()  # Set the model to evaluation mode

import matplotlib.pyplot as plt


def segment_image(image_path, model):
    # Define transformations to preprocess the input image
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((256, 256)),  # Resize to match the model's input size
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Load and preprocess the input image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform segmentation
    with torch.no_grad():
        output = model(image)

    # Convert the output to binary mask
    mask = torch.sigmoid(output) > 0.5  # Apply sigmoid and thresholding

    # Convert the mask tensor to numpy array
    mask = mask.squeeze(0).squeeze(0).numpy()  # Remove batch and channel dimensions, and convert to numpy

    return mask


# input_image_path = "input_image.jpg"
# segmented_mask = segment_image(input_image_path, model)


def visualize_segmentation(image_path, model):
    # Load the input image
    input_image = Image.open(image_path)

    # Segment the image
    segmented_mask = segment_image(image_path, model)

    # Plot original image and segmented mask side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot segmented mask
    axes[1].imshow(segmented_mask, cmap='gray')
    axes[1].set_title('Segmented Mask')
    axes[1].axis('off')

    plt.show()


# # Example usage
# input_image_path = "Dataset/benign/images/benign (26).png"
# visualize_segmentation(input_image_path, model)


def segment_images_in_folder(input_folder, output_folder, model):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Segment each image in the input folder
    for image_file in image_files:
        # Segment the image
        image_path = os.path.join(input_folder, image_file)
        mask = segment_image(image_path, model)

        # Save the segmented mask
        mask_image = Image.fromarray((mask * 255).astype('uint8'))
        mask_image.save(os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_mask.png"))



