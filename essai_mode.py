# Importing necessary libraries and modules
import warnings  # Import the 'warnings' module for handling warnings
warnings.filterwarnings("ignore")  # Ignore warnings during execution

import gc  # Import the 'gc' module for garbage collection
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
import itertools  # Import 'itertools' for iterators and looping
from collections import Counter  # Import 'Counter' for counting elements
import matplotlib.pyplot as plt  # Import Matplotlib for data visualization
from sklearn.metrics import (  # Import various metrics from scikit-learn
    accuracy_score,  # For calculating accuracy
    roc_auc_score,  # For ROC AUC score
    confusion_matrix,  # For confusion matrix
    classification_report,  # For classification report
    f1_score  # For F1 score
)

# Import custom modules and classes
from imblearn.over_sampling import RandomOverSampler # import RandomOverSampler
import accelerate # Import the 'accelerate' module
import evaluate  # Import the 'evaluate' module
from datasets import Dataset, Image, ClassLabel  # Import custom 'Dataset', 'ClassLabel', and 'Image' classes
from transformers import (  # Import various modules from the Transformers library
    TrainingArguments,  # For training arguments
    Trainer,  # For model training
    ViTImageProcessor,  # For processing image data with ViT models
    ViTForImageClassification,  # ViT model for image classification
    DefaultDataCollator  # For collating data in the default way
)
import torch  # Import PyTorch for deep learning
from torch.utils.data import DataLoader  # For creating data loaders
from torchvision.transforms import (  # Import image transformation functions
    CenterCrop,  # Center crop an image
    Compose,  # Compose multiple image transformations
    Normalize,  # Normalize image pixel values
    RandomRotation,  # Apply random rotation to images
    RandomResizedCrop,  # Crop and resize images randomly
    RandomHorizontalFlip,  # Apply random horizontal flip
    RandomAdjustSharpness,  # Adjust sharpness randomly
    Resize,  # Resize images
    ToTensor  # Convert images to PyTorch tensors
)
from PIL import ImageFile
from pathlib import Path
from tqdm import tqdm
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_path = Path(r'dataset\train')
train_image_dict = {}
train_file_names = []
train_labels = []

test_path = Path(r'dataset\test')
test_image_dict = {}
test_file_names = []
test_labels = []

for emotion_folder in train_path.glob('*'):
    if emotion_folder.is_dir():
        emotion = emotion_folder.name
        for file in emotion_folder.glob('*.*'):
            if file.is_file():
                train_labels.append(emotion)
                train_file_names.append(str(file))

for emotion_folder in test_path.glob('*'):
    if emotion_folder.is_dir():
        emotion = emotion_folder.name
        for file in emotion_folder.glob('*.*'):
            if file.is_file():
                test_labels.append(emotion)
                test_file_names.append(str(file))

# Print the total number of file names and labels
print(len(train_file_names), len(train_labels))
print(len(test_file_names), len(test_labels))

# Create a pandas dataframe from the collected file names and labels
df_train = pd.DataFrame.from_dict({"image": train_file_names, "label": train_labels})
print(df_train.shape)

print(df_train.head())

df_test = pd.DataFrame.from_dict({"image": test_file_names, "label": test_labels})
print(df_test.shape)

print(df_test.head())

# 'y' contains the target variable (label) we want to predict
y_train = df_train[['label']]
y_test = df_test[['label']]

# Drop the 'label' column from the DataFrame 'df' to separate features from the target variable
df_train = df_train.drop(['label'], axis=1)
df_test = df_test.drop(['label'], axis=1)

# Create a RandomOverSampler object with a specified random seed (random_state=83)
ros = RandomOverSampler(random_state=83)

# Use the RandomOverSampler to resample the dataset by oversampling the minority class
# 'df' contains the feature data, and 'y_resampled' will contain the resampled target variable
df_train, y_train_resampled = ros.fit_resample(df_train, y_train)
df_test, y_test_resampled = ros.fit_resample(df_test, y_test)

# Delete the original 'y' variable to save memory as it's no longer needed
del y_train
del y_test

# Add the resampled target variable 'y_resampled' as a new 'label' column in the DataFrame 'df'
df_train['label'] = y_train_resampled
df_test['label'] = y_test_resampled

# Delete the 'y_resampled' variable to save memory as it's no longer needed
del y_train_resampled
del y_test_resampled

# Perform garbage collection to free up memory used by discarded variables
gc.collect()

print(f"DF Train shape: {df_train.shape}")
print(f"DF Test shape: {df_test.shape}")

# Create a dataset from a Pandas DataFrame.
dataset_train = Dataset.from_pandas(df_train).cast_column("image", Image())
dataset_test = Dataset.from_pandas(df_test).cast_column("image", Image())

# Display the first image in the dataset
dataset_train[0]["image"]
dataset_test[0]["image"]

# The result will be a new list containing these elements.
train_labels_subset = train_labels[:5]
test_labels_subset = test_labels[:5]

# Printing the subset of labels to inspect the content.
print(f'Train labels : {train_labels_subset}')
print(f'Test labels : {train_labels_subset}')

# Create a list of unique labels by converting 'labels' to a set and then back to a list
train_labels_list = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
test_labels_list = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']

# Initialize empty dictionaries to map labels to IDs and vice versa
train_label2id, train_id2label = dict(), dict()
test_label2id, test_id2label = dict(), dict()

# Iterate over the unique labels and assign each label an ID, and vice versa
for i, label in enumerate(train_labels_list):
    train_label2id[label] = i
    train_id2label[i] = label

for i, label in enumerate(test_labels_list):
    test_label2id[label] = i
    test_id2label[i] = label

# Print the resulting dictionaries for reference
print("Mapping of IDs to Labels:", train_id2label, '\n')
print("Mapping of Labels to IDs:", train_label2id)

# Creating classlabels to match labels to IDs
TrainClassLabels = ClassLabel(num_classes=len(train_labels_list), names=train_labels_list)
TestClassLabels = ClassLabel(num_classes=len(test_labels_list), names=test_labels_list)

# Mapping labels to IDs
def map_train_label2id(example):
    example['label'] = TrainClassLabels.str2int(example['label'])
    return example

def map_test_label2id(example):
    example['label'] = TestClassLabels.str2int(example['label'])
    return example

dataset_train = dataset_train.map(map_train_label2id, batched=True)
dataset_test = dataset_test.map(map_test_label2id, batched=True)

# Casting label column to ClassLabel Object
dataset_train = dataset_train.cast_column('train_label', TrainClassLabels)
dataset_test = dataset_test.cast_column('test_label', TestClassLabels)

# Define the pre-trained ViT model string
model_str = "dima806/facial_emotions_image_detection"

# Create a processor for ViT model input from the pre-trained model
processor = ViTImageProcessor.from_pretrained(model_str)


# Retrieve the image mean and standard deviation used for normalization
image_mean, image_std = processor.image_mean, processor.image_std

# Get the size (height) of the ViT model's input images
size = processor.size["height"]
print("Size: ", size)

# Define a normalization transformation for the input images
normalize = Normalize(mean=image_mean, std=image_std)

# Define a set of transformations for training data
_train_transforms = Compose(
    [
        Resize((size, size)),             # Resize images to the ViT model's input size
        RandomRotation(90),               # Apply random rotation
        RandomAdjustSharpness(2),         # Adjust sharpness randomly
        RandomHorizontalFlip(0.5),        # Random horizontal flip
        ToTensor(),                       # Convert images to tensors
        normalize                         # Normalize images using mean and std
    ]
)

# Define a set of transformations for validation data
_val_transforms = Compose(
    [
        Resize((size, size)),             # Resize images to the ViT model's input size
        ToTensor(),                       # Convert images to tensors
        normalize                         # Normalize images using mean and std
    ]
)

# Define a function to apply training transformations to a batch of examples
def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Define a function to apply validation transformations to a batch of examples
def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

dataset_train.set_transform(train_transforms)

# Set the transforms for the test/validation data
dataset_test.set_transform(val_transforms)

# Define a collate function that prepares batched data for model training.
def collate_fn(examples):
    # Stack the pixel values from individual examples into a single tensor.
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    
    # Convert the label strings in examples to corresponding numeric IDs using label2id dictionary.
    labels = torch.tensor([example['test_label'] for example in examples])
    
    # Return a dictionary containing the batched pixel values and labels.
    return {"pixel_values": pixel_values, "labels": labels}

# Create a ViTForImageClassification model from a pretrained checkpoint with a specified number of output labels.
model = ViTForImageClassification.from_pretrained(model_str, num_labels=len(test_labels_list))

# Configure the mapping of class labels to their corresponding indices for later reference.
model.config.id2label = test_id2label
model.config.label2id = test_label2id

# Calculate and print the number of trainable parameters in millions for the model.
print(model.num_parameters(only_trainable=True) / 1e6)

# Load the accuracy metric from a module named 'evaluate'
accuracy = evaluate.load("accuracy")

# Define a function 'compute_metrics' to calculate evaluation metrics
def compute_metrics(eval_pred):
    # Extract model predictions from the evaluation prediction object
    predictions = eval_pred.predictions
    
    # Extract true labels from the evaluation prediction object
    label_ids = eval_pred.label_ids
    
    # Calculate accuracy using the loaded accuracy metric
    # Convert model predictions to class labels by selecting the class with the highest probability (argmax)
    predicted_labels = predictions.argmax(axis=1)
    
    # Calculate accuracy score by comparing predicted labels to true labels
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']
    
    # Return the computed accuracy as a dictionary with the key "accuracy"
    return {
        "accuracy": acc_score
    }

# Define the name of the evaluation metric to be used during training and evaluation.
metric_name = "accuracy"

# Define the name of the model, which will be used to create a directory for saving model checkpoints and outputs.
model_name = "facial_emotions_image_detection"

# Define the number of training epochs for the model.
num_train_epochs = 2

# Create an instance of TrainingArguments to configure training settings.
args = TrainingArguments(
    # Specify the directory where model checkpoints and outputs will be saved.
    output_dir=model_name,
    
    # Specify the directory where training logs will be stored.
    logging_dir='./logs',
    
    # Define the evaluation strategy, which is performed at the end of each epoch.
    evaluation_strategy="epoch",
    
    # Set the learning rate for the optimizer.
    learning_rate=1e-7,
    
    # Define the batch size for training on each device.
    per_device_train_batch_size=32,
    
    # Define the batch size for evaluation on each device.
    per_device_eval_batch_size=8,
    
    # Specify the total number of training epochs.
    num_train_epochs=num_train_epochs,
    
    # Apply weight decay to prevent overfitting.
    weight_decay=0.02,
    
    # Set the number of warm-up steps for the learning rate scheduler.
    warmup_steps=50,
    
    # Disable the removal of unused columns from the dataset.
    remove_unused_columns=False,
    
    # Define the strategy for saving model checkpoints (per epoch in this case).
    save_strategy='epoch',
    
    # Load the best model at the end of training.
    load_best_model_at_end=True,
    
    # Limit the total number of saved checkpoints to save space.
    save_total_limit=1,
    
    # Specify that training progress should not be reported.
    report_to="none"
)

trainer = Trainer(
    model,
    args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.evaluate()

trainer.train()

