from PIL import Image

import gradio as gr
import torch
import torchvision.transforms as transforms

from model import *

title = "Garment Classifier"
description = "Trained on the Fashion MNIST dataset (28x28 pixels). The model expects images containing only one garment article as in the examples."
inputs = gr.components.Image()
outputs = gr.components.Label()
examples = "examples"

model = torch.load("model/fashion.mnist.base.pt", map_location=torch.device("cpu"))

# Images need to be transformed to the `Fashion MNIST` dataset format
# see https://arxiv.org/abs/1708.07747
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalization
        transforms.Lambda(lambda x: 1.0 - x),  # Invert colors
        transforms.Lambda(lambda x: x[0]),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ]
)


def predict(img):
    img = transform(Image.fromarray(img))
    predictions = model.predictions(img)
    return predictions


with gr.Blocks() as demo:
    with gr.Tab("Garment Prediction"):
        gr.Interface(
            fn=predict,
            inputs=inputs,
            outputs=outputs,
            examples=examples,
            description=description,
        ).queue(default_concurrency_limit=5)

demo.launch()
