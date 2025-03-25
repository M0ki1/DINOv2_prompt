import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms

from  src.model_LN_prompt import Model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#loading de model
s3bir_model = Model()
model_ckp = torch.load("./models/model_flickr.ckpt", map_location=device, weights_only=False)
s3bir_model.load_state_dict(model_ckp['state_dict'])
s3bir_model.eval()
s3bir_model.to(device)

#image transformation
transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

#read image
imagepath = "./images/17516.jpg"
img = ImageOps.pad(Image.open(imagepath).convert('RGB'), (224, 224))
img = transform(img).to(device)

print(img.shape)
with torch.no_grad():
  image_features = s3bir_model(img.unsqueeze(0), dtype='image')
  # sketch_features = s3bir_model(img.unsqueeze(0), dtype='sketch')

#the feature vector
print(image_features.cpu().numpy())
