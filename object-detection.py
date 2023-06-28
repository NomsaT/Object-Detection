import numpy as np
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import requests
import cv2

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
path="dog.png" #can put your own image
image = Image.open(path)
# image = Image.open(requests.get(url, stream=True).raw)

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

# Convert PIL image to OpenCV format
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Draw bounding boxes on the image
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [int(i) for i in box.tolist()]
    cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(image_cv, f'{model.config.id2label[label.item()]}: {round(score.item(), 3)}',
                (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Convert the image back to PIL format for display
image_with_boxes = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
image_with_boxes.show()
