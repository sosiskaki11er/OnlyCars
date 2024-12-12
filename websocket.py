import asyncio
import websockets
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import io
import os
import json
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("server.log"),  # Log to a file named server.log
        logging.StreamHandler()  # Also output logs to the console
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 5 MB
byteBuffer = bytearray()

# Ensemble Model Definition
class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        # Model 1: ResNet50
        self.model1 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs1 = self.model1.fc.in_features
        self.model1.fc = nn.Linear(num_ftrs1, num_classes)

        # Model 2: EfficientNet B0
        self.model2 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs2 = self.model2.classifier[1].in_features
        self.model2.classifier[1] = nn.Linear(num_ftrs2, num_classes)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = (x1 + x2) / 2  # Average the outputs
        return x

# Load PyTorch model
def load_model(model_path, num_classes, device):
    model = EnsembleModel(num_classes=num_classes)
    try:
        logger.info(f"Loading model from {model_path}...")
        # Adjusted torch.load to specify weights_only=True if available
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            # For PyTorch versions that do not support weights_only
            state_dict = torch.load(model_path, map_location=device)
        # Adjust layers for the checkpoint's class count
        checkpoint_num_classes = state_dict["model1.fc.weight"].shape[0]
        logger.info(f"Checkpoint was trained with {checkpoint_num_classes} classes.")

        # Adjust ResNet50 classifier
        model.model1.fc = nn.Linear(model.model1.fc.in_features, checkpoint_num_classes)
        
        # Adjust EfficientNet classifier
        model.model2.classifier[1] = nn.Linear(model.model2.classifier[1].in_features, checkpoint_num_classes)

        # Load state dict
        model.load_state_dict(state_dict)
        logger.info("Model state dict loaded.")

        # Adjust layers back to the current number of classes if necessary
        if checkpoint_num_classes != num_classes:
            model.model1.fc = nn.Linear(model.model1.fc.in_features, num_classes)
            model.model2.classifier[1] = nn.Linear(model.model2.classifier[1].in_features, num_classes)
            logger.info(f"Adjusted model to {num_classes} classes.")
        else:
            logger.info("Model's output layer matches the number of classes.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

    model.to(device)
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")
    return model

# Preprocessing
def preprocess_image(image_stream, transform):
    try:
        logger.debug("Opening image...")
        image = Image.open(image_stream).convert('RGB')
        logger.debug("Image opened successfully.")
        tensor = transform(image).unsqueeze(0)
        logger.debug("Image transformed into tensor.")
        return tensor
    except UnidentifiedImageError:
        logger.error("Failed to identify image.")
        raise ValueError("Invalid or corrupted image.")
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise ValueError(f"Image preprocessing error: {e}")

# Transform definition
def get_transform(image_size=(299, 299)):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# WebSocket handler
async def handle_client(websocket, lock, model, transform, class_names):
    try:
        logger.info("Connection received from client.")

        # Receive binary data (image) from the client
        try:
            
            binary_data = await websocket.recv()
            logger.debug(f"Received {len(binary_data)} bytes of data from client.")
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Client disconnected before sending data: {e}")
            return  # Exit the handler gracefully

        # Validate image size
        if len(byteBuffer) > MAX_IMAGE_SIZE:
            error_msg = "Image size exceeds limit."
            logger.warning(error_msg)
            await websocket.send(json.dumps({"error": error_msg}))
            return
        byteBuffer.extend(binary_data)
        if len(binary_data)<1000000: 
         # Wrap binary data in BytesIO for processing
         image_stream = io.BytesIO(byteBuffer)
         byteBuffer.clear()
        
    

         # Preprocess the image
         logger.info("Preprocessing image...")
         input_tensor = preprocess_image(image_stream, transform)
         input_tensor = input_tensor.to(device)
         logger.info("Image preprocessed successfully.")

         # Make a prediction
         logger.info("Running model inference...")
         async with lock:  # Ensure thread safety for model inference
            outputs = await asyncio.get_event_loop().run_in_executor(
                None, model_forward, model, input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred = torch.max(probabilities, 1)
         logger.info("Model inference completed.")

         confidence_value = confidence.item()
         logger.debug(f"Confidence score: {confidence_value}")

         if confidence_value < 0.1:
            response = ""  # Send an empty string to the client
            logger.info("Confidence below threshold; sending empty response.")
         else:
            predicted_class = class_names[pred.item()]
            # Split the predicted class into company and model name
            if '_' in predicted_class:
                company, model_name = predicted_class.split('_', 1)
                model_name = model_name.replace('_', ' ')  # Replace underscores with spaces in the model name
            else:
                company = predicted_class
                model_name = ""
            # Format as "model_name company" with a space
            response = f"{model_name} {company}".strip()
            logger.info(f"Predicted class: {response}")

         # Send the prediction back to the client
         try:
            await websocket.send(response)
            logger.info("Response sent to client.")
         except Exception as e:
            logger.error(f"Error sending response to client: {e}")
            traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            logger.error(f"Traceback: {traceback_str}")

    except websockets.exceptions.ConnectionClosed as e:
        logger.warning(f"Client disconnected: {e}")
    except Exception as e:
        logger.error(f"Error in server: {e}")
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        logger.error(f"Traceback: {traceback_str}")
        # Send error message to client if possible
        try:
            await websocket.send(f"Server error: {str(e)}")
        except:
            pass  # Connection may already be closed

def model_forward(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs

# WebSocket server main
async def main():
    lock = asyncio.Lock()  # Thread safety for model inference

    # Start the server and pass model, transform, and class_names to handle_client
    server = await websockets.serve(
        lambda ws: handle_client(ws, lock, model, transform, class_names),
        HOST, PORT,
        max_size=None
        # compression=None  # Disable compression if client does not support it
    )
    logger.info("Server started and waiting for connections...")
    logger.info(f"Connect to the WebSocket at ws://{HOST}:{PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    # Parameters
    HOST = "172.20.10.11"  # Replace with your server's IP address
    PORT = 8765  # Replace with your desired port number
    model_path = "./output1/car_model_recognition_final.pt"  # Replace with your actual model path
    dataset_folder = "./car_models-master"  # Replace with your dataset folder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size=(299, 299))

    # Load class names from the dataset folder
    if os.path.exists(dataset_folder):
        logger.info(f"Loading class names from dataset folder {dataset_folder}...")
        class_names = sorted([
            d.name for d in os.scandir(dataset_folder) if d.is_dir()
        ])
    else:
        logger.error(f"Dataset folder not found at {dataset_folder}.")
        class_names = []

    # Ensure the class count matches the model
    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError("No class names found in dataset folder. Please provide a valid dataset.")
    else:
        logger.info(f"{num_classes} class names loaded.")

    # Load the model
    model = load_model(model_path, num_classes, device)

    # Start the WebSocket server
    logger.info(f"Starting WebSocket server at ws://{HOST}:{PORT}")
    asyncio.run(main())
