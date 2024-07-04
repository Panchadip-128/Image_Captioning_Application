import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the models and processor
processor_base = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

processor_large = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Define the function for unconditional image captioning
def caption_image_unconditional(image):
    inputs = processor_base(image, return_tensors="pt")
    outputs = model_base.generate(**inputs)
    caption = processor_base.decode(outputs[0], skip_special_tokens=True)
    return caption

# Define the function for conditional image captioning
def caption_image_conditional(image, text):
    inputs = processor_large(image, text, return_tensors="pt")
    outputs = model_large.generate(**inputs)
    caption = processor_large.decode(outputs[0], skip_special_tokens=True)
    return caption
