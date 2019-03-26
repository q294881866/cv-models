# Import needed packages
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
import requests
import shutil
from io import open
import os, shutil
from PIL import Image
import json
print("Prediction in progress")

model = torch.load("/Users/peipengfei/Downloads/gender.pkl")
print(model)
model.eval()
img_to_tensor = transforms.ToTensor()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict_image(image_path):
    img = Image.open(image_path)
    # plt.imshow(image)
    # plt.show()
    # img = image.resize((224, 224))
    # Preprocess the image

    image_tensor = img_to_tensor(img)
    image_tensor = image_tensor.resize_(1, 3, 224, 224)

    model.to(device)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)
    print(output)
    index = output.data.numpy().argmax()

    return index


def img_list(dirname):
    for filename in os.listdir(dirname):
        path = dirname + filename
        if os.path.isdir(path):
            path += '/'
            img_list(path)
        else:
            try:
                if path.endswith(".png"):
                    index = predict_image(path)
                    fpath, fname = os.path.split(path)
                    if 1 == index:
                        dstfile = os.path.join("/Users/peipengfei/Downloads/face/female", fname)
                        shutil.move(path, dstfile)
                    elif 0 == index:
                        dstfile = os.path.join("/Users/peipengfei/Downloads/face/male", fname)
                        shutil.move(path, dstfile)

            except:
                print(path)

dirname = "/Users/peipengfei/Downloads/faceDB/"
img_list(dirname)
print("----end")