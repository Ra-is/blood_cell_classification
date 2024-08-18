# Blood Cell Classification Project

## Overview

This project aims to classify blood cell subtypes using various machine learning models. The dataset used for this project was sourced from Kaggle, specifically the [Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) dataset. The diagnosis of blood-based diseases often involves identifying and characterizing patient blood samples, and this project explores automated methods to detect and classify blood cell subtypes, which have important medical applications.

## Dataset

The dataset contains a total of 12,500 augmented images of blood cells across four different classes:

- **Eosinophil**
- **Lymphocyte**
- **Monocyte**
- **Neutrophil**

### Dataset Structure

- **Training Set**: 9,957 images
- **Validation Set**: 2,487 images

Each image is labeled according to its cell type. The dataset also includes an additional set of 410 original images with bounding boxes and subtype labels.

### Data Loading

The datasets are loaded using TensorFlow's `image_dataset_from_directory` function:

```python
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Load training dataset
train_dataset = image_dataset_from_directory(
    train_directory,
    image_size=(224, 224),
    label_mode="categorical",
    batch_size=32,
    shuffle=True,
)

# Load validation dataset
val_dataset = image_dataset_from_directory(
    val_directory,
    image_size=(224, 224),
    label_mode="categorical",
    batch_size=32,
    shuffle=True,
)
```

## Approaches

### 1. Convolutional Neural Network (CNN)

A custom Convolutional Neural Network was built from scratch to classify the blood cell images into the four categories. The architecture consists of multiple convolutional layers followed by max-pooling and fully connected layers. The model was trained using the Adam optimizer and categorical cross-entropy loss function.

### 2. Pretrained EfficientNet Model

This approach leverages the pretrained EfficientNet model, known for its efficiency and performance in image classification tasks. The model was fine-tuned on the blood cell dataset by replacing the top layers with a custom classifier and training on our specific classes.

### 3. Pretrained Vision Transformer (ViT)

The Vision Transformer (ViT) model from Hugging Face was utilized for this approach. ViT processes images by dividing them into patches and treating them similarly to tokens in NLP tasks. The pretrained model was fine-tuned on our dataset to adapt it to the blood cell classification task.

## Results

Each model was evaluated using metrics such as accuracy. The results are as follows:

| Model                  | Accuracy |
| ---------------------- | -------- |
| **CNN**                | 25.2%    |
| **EfficientNet**       | 64.5%    |
| **Vision Transformer** | 76.8%    |

The Vision Transformer model achieved the highest accuracy, demonstrating the effectiveness of transformer-based architectures in image classification tasks.

**Note:** The models were not trained for an extended period, which may have limited their accuracy. It is likely that training the models for longer durations could yield higher accuracy and improved performance overall.
