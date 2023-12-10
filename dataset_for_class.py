from PIL import Image, ImageEnhance, ImageFilter
import os
import numpy as np
import cv2
import random

# Input and output directories
input_dir = r"C:\Users\render\Desktop\puncher\makin_dataset\all_cards_w_names_jpg"
output_dir = r"C:\Users\render\Desktop\puncher\makin_dataset\dataset_for_class"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create folders for ranks and suits inside the output directory
ranks_dir = os.path.join(output_dir, 'ranks')
suits_dir = os.path.join(output_dir, 'suits')
os.makedirs(ranks_dir, exist_ok=True)
os.makedirs(suits_dir, exist_ok=True)

# List of ranks and suits
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace', 'joker']
suits = ['hearts', 'spades', 'clubs', 'diamonds', 'joker']

# Padding size
padding_size = 2
# Number of repetitions for each picture
num_repetitions = 50
# Split ratio for training and validation
split_ratio = 0.1
split_counter = int(1 / split_ratio)

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)

    # Check if the file is a JPG image
    if filename.lower().endswith(('.jpg', '.jpeg')):
        for i in range(num_repetitions):
            # Open the image
            img = Image.open(input_path)

            # Get the original image size
            original_size = img.size

            # Calculate the new size with padding
            new_size = (original_size[0] * (padding_size + 1), original_size[1] * (padding_size + 1))

            # Create a new RGBA image with 100% transparency padding
            img_with_padding = Image.new('RGBA', new_size, (255, 255, 255, 0))

            # Paste the original image onto the center of the new image
            offset = ((new_size[0] - original_size[0]) // 2, (new_size[1] - original_size[1]) // 2)
            img_with_padding.paste(img, offset)

            # Randomly rotate in x, y, and z axes by a random angle in the range [0, 90] degrees
            angles = [random.uniform(0, 360) for _ in range(3)]
            img_rotated = img_with_padding.rotate(angles[0], resample=Image.BICUBIC, center=(new_size[0] // 2, new_size[1] // 2))
            img_rotated = img_rotated.rotate(angles[1], resample=Image.BICUBIC, center=(new_size[0] // 2, new_size[1] // 2))
            img_rotated = img_rotated.rotate(angles[2], resample=Image.BICUBIC, center=(new_size[0] // 2, new_size[1] // 2))

            # Convert PIL image to numpy array
            img_array = np.array(img_rotated)

            # Apply random perspective transformation using opencv
            rows, cols, _ = img_array.shape

            # Define the four corners of the original image
            pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])

            # Define the four corners of the transformed image with random perspective factors
            perspective_factors = [random.uniform(0, 0.7) for _ in range(4)]
            perspective_pts = np.float32([
                [perspective_factors[0] * (cols - 1), 0],
                [cols - 1, 0],
                [0, rows - 1],
                [cols - 1 - perspective_factors[1] * (cols - 1), rows - 1]
            ])

            # Calculate the perspective transformation matrix
            perspective_matrix = cv2.getPerspectiveTransform(pts1, perspective_pts)

            # Apply the perspective transformation
            img_transformed = cv2.warpPerspective(img_array, perspective_matrix, (cols, rows))

            # Convert the result back to PIL image
            img_transformed = Image.fromarray(img_transformed, 'RGBA')

            # Crop the image by removing fully transparent rows and columns
            img_cropped = img_transformed.crop(img_transformed.getbbox())

            # Generate random values for contrast, brightness, sharpness, color, saturation, blur, and gamma correction
            contrast_factor = random.uniform(0.5, 1.5)
            brightness_factor = random.uniform(-0.2, 0.2)
            sharpness_factor = random.uniform(0, 100)
            color_factor = random.uniform(0.5, 1.5)
            saturation_factor = random.uniform(0.5, 1.5)
            blur_radius = random.uniform(0, 20)
            gamma_factor = random.uniform(0.5, 1.5)

            # Adjust contrast
            enhancer = ImageEnhance.Contrast(img_cropped)
            img_cropped = enhancer.enhance(contrast_factor)

            # Adjust brightness
            enhancer = ImageEnhance.Brightness(img_cropped)
            img_cropped = enhancer.enhance(1 + brightness_factor)

            # Adjust sharpness
            enhancer = ImageEnhance.Sharpness(img_cropped)
            img_cropped = enhancer.enhance(sharpness_factor)

            # Adjust color balance
            enhancer = ImageEnhance.Color(img_cropped)
            img_cropped = enhancer.enhance(color_factor)

            # Adjust saturation
            enhancer = ImageEnhance.Color(img_cropped)
            img_cropped = enhancer.enhance(saturation_factor)

            # Apply blur
            img_cropped = img_cropped.filter(ImageFilter.GaussianBlur(blur_radius))

            # Apply gamma correction
            img_cropped = ImageEnhance.Brightness(img_cropped).enhance(gamma_factor)

            # Convert 'RGBA' to 'RGB' and fill the alpha channel with color
            random_r = random.randint(0, 255)
            random_g = random.randint(0, 255)
            random_b = random.randint(0, 255)
            img_cropped = Image.alpha_composite(Image.new('RGBA', img_cropped.size, (random_r, random_g, random_b, 255)), img_cropped)

            # Randomly resize the image from 50% to 100%
            resize_factor = random.uniform(0.5, 1)
            new_width = int(img_cropped.width * resize_factor)
            new_height = int(img_cropped.height * resize_factor)
            img_cropped = img_cropped.resize((new_width, new_height), resample=Image.BICUBIC)

            # Convert the image to 'RGB' mode before saving as JPEG
            img_cropped = img_cropped.convert('RGB')

            # Save the final image with a JPG extension to the output directory
            output_filename = f"{os.path.splitext(filename)[0]}_transformed_cropped_{i + 1}.jpg"

            # Determine rank and suit from the original filename
            rank, suit = os.path.splitext(filename)[0].split('_')
            if rank == 'joker':
                suit = 'joker'
            rank_val_folder = os.path.join(ranks_dir, 'val', rank)
            suit_val_folder = os.path.join(suits_dir, 'val', suit)
            rank_train_folder = os.path.join(ranks_dir, 'train', rank)
            suit_train_folder = os.path.join(suits_dir, 'train', suit)

            # Save the transformed image into the appropriate folders
            os.makedirs(rank_val_folder, exist_ok=True)
            os.makedirs(suit_val_folder, exist_ok=True)
            os.makedirs(rank_train_folder, exist_ok=True)
            os.makedirs(suit_train_folder, exist_ok=True)

            if (i + 1) % split_counter == 0:
                output_rank_path = os.path.join(rank_val_folder, output_filename)
                output_suit_path = os.path.join(suit_val_folder, output_filename)
            else:
                output_rank_path = os.path.join(rank_train_folder, output_filename)
                output_suit_path = os.path.join(suit_train_folder, output_filename)

            img_cropped.save(output_rank_path, format='JPEG')
            img_cropped.save(output_suit_path, format='JPEG')

            print(f"Converted {filename} to {output_filename}")


print(f'''Conversion with padding, random rotations (0 to 360 degrees), 
      random perspective transformations (0 to 0.7), random contrast, 
      brightness adjustments, and cropping ({num_repetitions} times) complete.''')