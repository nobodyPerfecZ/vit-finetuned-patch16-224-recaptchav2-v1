# Finetuned Vision Transformer

This repository contains a Vision Transformer (ViT) model fine-tuned on the ReCAPTCHAv2-29k dataset.
The dataset comprises 29,568 labeled images spanning 5 classes, each resized to a resolution of 224×224 pixels.

## Model description

This model builds on a pre-trained ViT backbone and is fine-tuned on the ReCAPTCHAv2-29k dataset.
It leverages the transformer-based architecture to capture global contextual information effectively, making it well-suited for tasks with diverse visual patterns like ReCAPTCHA classification.

## Intended uses & limitations

This fine-tuned ViT model is designed for multi-label-classification tasks involving ReCAPTCHA-like visual patterns.
Potential applications include:

- Automated ReCAPTCHA analysis for research or accessibility tools
- Benchmarking and evaluation of ReCAPTCHA-solving models
- Educational purposes, such as studying transformer behavior on visual data

The model is particularly useful in academic and experimental contexts where understanding transformer-based classification on noisy or distorted visual data is a priority.

## How to use

Here is how to use this model to classify an image of the ReCAPTCHAv2-29k dataset into one of the 5 classes:

```python
import requests
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

url = "https://raw.githubusercontent.com/nobodyPerfecZ/recaptchav2-29k/refs/heads/master/data/bicycle/bicycle_0.png"
image = Image.open(requests.get(url, stream=True).raw)
processor = ViTImageProcessor.from_pretrained(
    "nobodyPerfecZ/vit-finetuned-patch16-224-recaptchav2-v1"
)
model = ViTForImageClassification.from_pretrained(
    "nobodyPerfecZ/vit-finetuned-patch16-224-recaptchav2-v1"
)
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 5 classes
predictions = (outputs.logits >= 0.5).to(int)
predicted_class_indicies = torch.where(predictions == 1)[1]
labels = [model.config.id2label[idx.item()] for idx in predicted_class_indicies]
print(f"Predicted labels: {labels}")
```

## Training data

The ViT model was fine-tuned on [ReCAPTCHAv2-29k dataset](https://huggingface.co/datasets/nobodyPerfecZ/recaptchav2-29k), a dataset consisting of 29.568 images and 5 classes.

## Training procedure

### Preprocessing

The exact details of preprocessing of images during training/validation can be found [here](https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py).

Images are resized/rescaled to the same resolution (224x224) and normalized across the RGB channels with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).

## Evaluation results

The ViT model was evaluated on a held-out test set from the ReCAPTCHAv2-29k dataset.
Two key metrics were used to assess performance:

| Metric           | Score |
| ---------------- | ----- |
| Top-1 Accuracy   | 0.93  |
| Hamming Accuracy | 0.97  |

- Top-1 Accuracy reflects the proportion of images where the model's most confident prediction matched the true label
- Hamming Accuracy measures the fraction of correctly predicted labels per sample

These results indicate strong classification performance, especially given the visual complexity and distortion typical in ReCAPTCHA images.
