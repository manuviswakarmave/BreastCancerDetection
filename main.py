
import io

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
from PIL import Image
import base64

import segmentation
import segmentation_models_pytorch as smp

from prediction import predict

# Load the trained segmentation model model
segmentation_model = smp.Unet('resnet34', encoder_weights=None, in_channels=1,
                              classes=1)  # Define the same architecture as during training
segmentation_model.load_state_dict(
    torch.load('segmentation_model.pth', map_location=torch.device('cpu')))  # Load the trained weights
segmentation_model.eval()  # Set the model to evaluation mode

app = FastAPI()
prediction_router = APIRouter(prefix="/BreastCancerPrediction", tags=["Prediction Controller"])


# Define class for JSON response
class PredictionResponse:
    def __init__(self, original_image_base64: str, processed_image_base64: str, predicted_class: str):
        self.original_image_base64 = original_image_base64
        self.processed_image_base64 = processed_image_base64
        self.predicted_class = predicted_class


# Define FastAPI endpoint to accept image file
@prediction_router.post("/prediction_workflow/")
async def prediction(image: UploadFile = File(...)):
    # Save uploaded image to temporary file
    with NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp.write(image.file.read())
        temp.flush()

        # Open the image using PIL
        img = Image.open(temp.name)

        # perform segmentation
        segmented_mask = segmentation.segment_image(img, segmentation_model)
        segmented_image = Image.fromarray(segmented_mask)

        # perfrom classification
        predicted_class = predict(segmented_image.convert('RGB'))

        original_image_bytes = io.BytesIO()
        img.save(original_image_bytes, format='PNG')
        original_image_bytes = original_image_bytes.getvalue()

        processed_image_bytes = io.BytesIO()
        segmented_image.save(processed_image_bytes, format='PNG')
        processed_image_bytes = processed_image_bytes.getvalue()

        # Convert images to base64 strings
        original_image_base64 = base64.b64encode(original_image_bytes).decode("utf-8")
        processed_image_base64 = base64.b64encode(processed_image_bytes).decode("utf-8")

        if predicted_class == 0:
            predicted_class = "Tumour Detected"
        else:
            predicted_class = "Normal Condition"

    # Create JSON response object
    response_data = PredictionResponse(original_image_base64, processed_image_base64, predicted_class)

    # Return prediction as JSON response
    return JSONResponse(content=response_data.__dict__)


app.include_router(prediction_router)
if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True, workers=2)
