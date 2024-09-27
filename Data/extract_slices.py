import imageio
import numpy as np
from PIL import Image
import os

# 1. Import a volumetric image (it will be around 2000 slices)
def load_volumetric_image(filepath):
    # Load the volumetric image (assuming it's a tiff stack or similar format)
    vol_image = imageio.volread(filepath)
    print(f"Loaded image with shape: {vol_image.shape}")
    return vol_image

# 2. Take the range of the slices you are interested in
def get_slice_range(vol_image, start_slice, end_slice):
    return vol_image[start_slice:end_slice], list(range(start_slice, end_slice))

# 3. Apply the display range you specify (no normalization to 0-255)
def apply_display_range(slice_img, display_min, display_max):
    # Clip values outside the specified range
    slice_img_display = np.clip(slice_img, display_min, display_max)
    # Return the image with original intensities but clipped for display purposes
    return slice_img_display

# 4. Extract 3 images in even distances within this range
def extract_even_slices(slices, slice_numbers, num_slices=3):
    step = len(slices) // num_slices
    extracted_slices = [(slices[i * step], slice_numbers[i * step]) for i in range(num_slices)]
    return extracted_slices

# 5. Save as tif and jpeg, using the actual slice number in the filename
def save_images(extracted_slices, output_dir, display_min, display_max, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for slice_img, slice_num in extracted_slices:
        # Apply the display range (clipping values) for visualization
        img_display = apply_display_range(slice_img, display_min, display_max)
        
        # Convert to PIL image without scaling, keep original values
        img = Image.fromarray(img_display)
        
        # Save as TIFF with original values
        tiff_path = os.path.join(output_dir, f"{filename}_sl_{slice_num}.tif")
        img.save(tiff_path, format='TIFF')

        # Save as JPEG with original values
        # jpeg_path = os.path.join(output_dir, f"{filename}_sl_{slice_num}.jpeg")
        # img.save(jpeg_path, format='JPEG')
        
        print(f"Saved slice {slice_num} as TIFF with display range {display_min}-{display_max}")

# Main execution
def process_volumetric_image(filepath, start_slice, end_slice, output_dir, display_min, display_max):
    # Step 1: Load the volumetric image
    vol_image = load_volumetric_image(filepath)
    
    # Step 2: Get the range of slices and their corresponding slice numbers
    slices_of_interest, slice_numbers = get_slice_range(vol_image, start_slice, end_slice)
    
    # Step 3: Extract 3 images at even distances within this range, including the actual slice number
    extracted_slices = extract_even_slices(slices_of_interest, slice_numbers)
    
    # Step 4: Save the extracted slices as both TIFF and JPEG
    filename = os.path.basename(filepath)
    save_images(extracted_slices, output_dir, display_min, display_max, filename)

# Example execution for multiple images
if __name__ == "__main__":
    image_list = [
        ("CEP_318_2022_XR.tif", (715, 881)),
        ("CEP_322_2023_XR.tif", (685, 859)),
        ("CEP_323_2024_XR.tif", (453, 488)),
        ("CEP_330_2022_XR.tif", (774, 787)),
        ("CEP_335_2023_XR.tif", (879, 1054)),
        ("CEP_1181_2024_XR.tif", (401, 437)),
        ("CEP_1186A_2022_XR.tif", (983, 1034)),
        ("CEP_1189_2023_XR.tif", (838, 936)),
        ("CEP_1191_2024_XR.tif", (966, 1108)),
        ("CEP_1193_2022_XR.tif", (602, 705)),
        ("CEP_1195_2023_XR.tif", (1033, 1136)),
        ("CEP_313B_2024_XR.tif", (452, 639)),
        ("CEP_988B_2022_XR.tif", (578, 653)),
        ("CEP_1266A_2023_XR.tif", (656, 813)),
        ("CEP_368B_2024_XR.tif", (732, 1024)),
        ("CEP_380A_2022_XR.tif", (822, 1171)),
        ("CEP_378A_2023_XR.tif", (1131, 1225)),
        ("CEP_378B_2024_XR.tif", (997, 1166)),
        ("CEP_764B_2022_XR.tif", (606, 910)),
        ("CEP_2184A_2023_XR.tif", (524, 573)),
    ]
    
    filepath = "/home/phukon/Desktop/Results_2024/Cep_Monitoring/XR/4_Registered_Images/16bit_Registered_2022_2023_2024/"
    output_dir = "/home/phukon/Desktop/Model_Fitting/new_dataset"  # Directory where images will be saved

    # Process each image in the list
    for filename, (start_slice, end_slice) in image_list:
        process_volumetric_image(filepath + filename, start_slice, end_slice, output_dir, 40, 1500)
