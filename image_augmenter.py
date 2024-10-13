#### Import necessary libraries
import os
import cv2
import albumentations as A
from tqdm import tqdm

#### Path and augmentation configuration
# Input and output directories
INPUT_DIRECTORY = 'dataset'
OUTPUT_DIRECTORY = INPUT_DIRECTORY + '_augmented'

# Augmentation parameters
ROTATE_LIMIT = 60 
SHIFT_LIMIT = 0.2  
SCALE_LIMIT = 0.2
BRIGHTNESS_LIMIT = 0.2 
CONTRAST_LIMIT = 0.4
HORIZONTAL_FLIP = True  
AUGMENTATION_COUNT = 2

#### Main script
# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

# Define augmentation pipeline
augmentations = A.Compose([
    A.Rotate(limit=ROTATE_LIMIT, p=0.5),
    A.ShiftScaleRotate(shift_limit=SHIFT_LIMIT, scale_limit=SCALE_LIMIT, rotate_limit=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=BRIGHTNESS_LIMIT, contrast_limit=CONTRAST_LIMIT, p=0.5),
    A.HorizontalFlip(p=HORIZONTAL_FLIP),
])

def augment_image(image, augment_count):
    """
    Augments an image multiple times.
    :param image: Input image
    :param augment_count: Number of augmentations to apply
    :return: List of augmented images
    """
    augmented_images = []
    for _ in range(augment_count):
        augmented = augmentations(image=image)['image']
        augmented_images.append(augmented)
    return augmented_images

def augment_dataset():
    """
    Augments all images in subdirectories of the input directory and saves them to respective subdirectories in the output directory.
    """
    # Iterate through the subdirectories in the input directory
    for subdir, _, files in os.walk(INPUT_DIRECTORY):
        relative_subdir = os.path.relpath(subdir, INPUT_DIRECTORY)
        output_subdir = os.path.join(OUTPUT_DIRECTORY, relative_subdir)

        # Create corresponding subdirectory in the output directory
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        print(f"Augmenting images in {subdir}...")

        # Iterate through all images in the current subdirectory
        for filename in tqdm(files):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                filepath = os.path.join(subdir, filename)
                image = cv2.imread(filepath)

                if image is None:
                    print(f"Error reading image: {filepath}")
                    continue

                # Save the original image in the output subdirectory
                original_filename = f"{os.path.splitext(filename)[0]}_original.png"
                original_filepath = os.path.join(output_subdir, original_filename)
                cv2.imwrite(original_filepath, image)

                # Augment the image
                augmented_images = augment_image(image, AUGMENTATION_COUNT)

                # Save each augmented image in the respective output subdirectory
                for i, aug_image in enumerate(augmented_images):
                    aug_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.png"
                    aug_filepath = os.path.join(output_subdir, aug_filename)
                    cv2.imwrite(aug_filepath, aug_image)

if __name__ == '__main__':
    augment_dataset()