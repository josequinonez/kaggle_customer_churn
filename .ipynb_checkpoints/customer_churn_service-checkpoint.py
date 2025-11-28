import bentoml
import numpy as np
from typing import List
import sys 
import pandas as pd
import torch
import logging
import os

# Configure basic Python logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the service as a class with the bentoml.service decorator
@bentoml.service(name="churn_prediction_service")
class ChurnPredictionService:
    """
    BentoML Service for predicting customer churn using a PyTorch model and an Scikit-learn preprocessor.
    """

    def __init__(self):
        # The directory containing customer_churn_service.py and model_architecture.py
        model_architecture_dir = os.getcwd()
        # --- CRITICAL FIX: Add model directory to sys.path ---
        if model_architecture_dir not in sys.path:
            sys.path.append(model_architecture_dir)
            logger.info(f"Added model directory to sys.path: {model_architecture_dir}")
        # -----------------------------------------------------

        # Load the latest version of the 'Churn Prediction Model'
        try:            
            # 1. Get the model artifact from the BentoML Store
            model_tag = "churn_prediction_model:latest"
            bento_model = bentoml.models.get(model_tag)

            # 2. Load the model using the built-in PyTorch loader
            self.loaded_model = bentoml.pytorch.load_model(bento_model, weights_only=False) # Disables strict security checks for trusted files
            
            # Set model to evaluation mode
            self.loaded_model.eval() # Set model to evaluation mode once loaded
            self.loaded_model.to('cpu')
            logger.info(f"PyTorch Model loaded from BentoML store: {model_tag}")

        except Exception as e:
            logger.error(f"Error loading Churn Prediction Model: {e}")
            self.loaded_model = None

        # Load the latest version of the 'Churn Preprocessor'
        try:
            preprocessor_tag = "churn_preprocessor:latest"
            bento_preprocessor = bentoml.models.get(preprocessor_tag)
            
            # Load the preprocessor using the built-in Scikit-Learn loader
            self.loaded_preprocessor = bentoml.sklearn.load_model(bento_preprocessor)
            logger.info(f"Preprocessor loaded from BentoML store: {preprocessor_tag}")

        except Exception as e:
            logger.error(f"Error loading Churn Preprocessor: {e}")
            self.loaded_preprocessor = None

    # Define the API endpoint to directly accept a JSON list of dictionaries
    @bentoml.api
    def predict(self, input_data: dict) -> List[int]:
        if self.loaded_model is None or self.loaded_preprocessor is None:
            raise RuntimeError("Models not loaded. Cannot make predictions.")

        # Log a custom event using the standard Python logger
        logger.info(f"Received input data: {str(input_data)[:200]}...")

        # Convert list of dictionaries to a pandas DataFrame
        input_features = input_data.get('data')
        input_df = pd.DataFrame(input_features)

        # Preprocess the input data using the loaded preprocessor
        processed_data = self.loaded_preprocessor.transform(input_df)

        # Convert processed data to PyTorch tensor
        device = torch.device("cpu") 
        input_tensor = torch.tensor(processed_data, dtype=torch.float32).to(device)

        # Make predictions with the loaded model
        with torch.no_grad():
            # Since the model already includes Sigmoid, the output is probability [0, 1]
            outputs = self.loaded_model(input_tensor) 
                
            # Convert tensor output to NumPy array
            probabilities = outputs.cpu().numpy().flatten()
                
            # Convert probabilities to binary predictions (0 or 1)
            predictions = (probabilities > 0.5).astype(int).tolist()

            #predictions = (outputs > 0.5).float().cpu().numpy()
            #predictions = (outputs > 0.5).int().cpu()

        # Log data for monitoring purposes
        with bentoml.monitor("churn_prediction") as monitor:
            monitor.log(data=str(input_data), name="input_data", role="feature", data_type="categorical")
            monitor.log(data=str(predictions), name="prediction_output", role="prediction", data_type="categorical")

        logger.info(f"Generated predictions: {predictions}")

        # Convert numpy array predictions back to a list for JSON output
        return predictions