import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation,Trainer, TrainingArguments


class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations_file):
        self.images_dir = "/home/phukon/Desktop/Annotation_VIA/"
        self.annotations_file = "/home/phukon/Desktop/Annotation_VIA/Train/via_project_2Sep2024_16h21m_json.json"
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
        with open(annotations_file) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.image_filenames)

    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        image = Image.open(img_path)
        annotations = self.annotations[self.image_filenames[idx]]

        labels = annotations["Tissue class"]

        # Use feature extractor to prepare the image
        encoding = self.feature_extractor(images=image, return_tensors="pt")

        # Prepare the labels
        # You may need to process and format labels correctly
        return {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'labels': torch.tensor(labels)
        }
    
# Define paths
images_dir = '/home/phukon/Desktop/Annotation_VIA/'
annotations_file = '/home/phukon/Desktop/Annotation_VIA/Train/via_project_2Sep2024_16h21m_json.json'

# Load the feature extractor and model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Prepare dataset and dataloader
dataset = CustomDataset(images_dir, annotations_file)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    output_dir='./results',
    save_steps=10_000,
    save_total_limit=2
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset  # You should have a separate validation dataset in a real scenario
)

# Train the model
trainer.train()