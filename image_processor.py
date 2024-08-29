import cv2
import numpy as np
from segment_anything import SamPredictor  # Adjust import based on your actual library
from ultralytics import YOLO

class ImageProcessor:
    def __init__(self, yolov8_model_path, sam_model_path):
        self.model = YOLO(yolov8_model_path)
        self.predictor = SamPredictor(sam_model_path)
    
    def process_image(self, image_path):
        # Detect objects in the image
        results = self.model.predict(source=image_path, conf=0.25, classes=0)
        
        # Load and prepare the image
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        # Initialize a white background
        white_background = np.ones_like(image) * 255
        
        # Create an empty mask for accumulating all individual masks
        combined_binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Iterate through each detected object
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                bbox = box.xyxy.tolist()
                input_box = np.array(bbox)
                
                # Set the image for the predictor
                self.predictor.set_image(image)
                
                # Predict the mask for the current bounding box
                masks, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                # Convert the mask to a binary mask and accumulate it
                segmentation_mask = masks[0]
                binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
                combined_binary_mask = np.maximum(combined_binary_mask, binary_mask)
        
        # Apply the combined mask to the white background and the original image
        new_image = white_background * (1 - combined_binary_mask[..., np.newaxis]) + image * combined_binary_mask[..., np.newaxis]
        
        # Save the final image
        new_image_path = image_path.replace('uploads', 'uploads/processed')
        cv2.imwrite(new_image_path, cv2.cvtColor(new_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        return new_image_path
