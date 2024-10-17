import os

# Define the directories
image_dir = "/home/phukon/Desktop/Model_Fitting/weka_dataset/images/train_set/"
mask_dir = "/home/phukon/Desktop/Model_Fitting/weka_dataset/masks/"

# List all image filenames
image_filenames = os.listdir(image_dir)

for img_filename in image_filenames:
    # Extract the base name without the extension
    base_name = os.path.splitext(img_filename)[0]
    
    # Create the new mask filename
    new_mask_filename = f"{base_name}_mask.tif"  # Adjust suffix as needed
    
    # Construct full paths
    old_mask_path = os.path.join(mask_dir, f"mask{base_name[-1]}.tif")  # Example: map 'mask1.tif' to 'image1.tif'
    new_mask_path = os.path.join(mask_dir+"/test/", new_mask_filename)
    
    # Check if the old mask exists before renaming
    if os.path.exists(old_mask_path):
        # Rename the file
        os.rename(old_mask_path, new_mask_path)
        print(f'Renamed: {old_mask_path} to {new_mask_path}')
    else:
        print(f'Old mask not found: {old_mask_path}')