import os
import json
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class SegmentationDataset(Dataset):
    def __init__(self, json_file, image_dir, transform_mask=None, transform_image=None):
        # Load the JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image information
        image_info = list(self.data.values())[idx]
        image_filename = image_info['filename']
        image_path = os.path.join(self.image_dir, image_filename)
        regions = image_info['regions']

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Create an empty mask with class values
        mask = Image.new('I', image.size, 0)  # 'I' mode for 32-bit integer pixels

        for region in regions:
            shape = region['shape_attributes']
            tissue_class = region['region_attributes'].get('Tissue Class', 'Background')
            class_index = self.class_to_index(tissue_class)

            if shape['name'] == 'polygon':
                points = list(zip(shape['all_points_x'], shape['all_points_y']))
                self.draw_polygon(mask, points, class_index)
            elif shape['name'] == 'rect':
                x, y, width, height = shape['x'], shape['y'], shape['width'], shape['height']
                points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
                self.draw_polygon(mask, points, class_index)

        # Convert mask to numpy array for consistency with image transformations
        mask = np.array(mask, dtype=np.int32)

        if self.transform_image:
            image = self.transform_image(image)
            # Custom transformation to ensure mask is handled correctly
            mask = self.transform_mask(Image.fromarray(mask))

        return image, mask

    def draw_polygon(self, mask, points, class_index):
        # Draw a polygon on the mask using class index
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, outline=class_index, fill=class_index)

    def class_to_index(self, tissue_class):
        # Map tissue classes to integer values
        class_map = {
            'Background': 0,
            'Healthy Functional': 1,
            'Healthy NonFunctional': 2,
            'Necrotic Infected': 3,
            'Necrotic Dry': 4,
            'Bark': 5
        }
        return class_map.get(tissue_class, 0)  # Default to Background if class not found
