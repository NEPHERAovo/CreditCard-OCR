import cv2
import os
import random

def process(num_iterations,original_dir_path,dir_path):
    for i in range(num_iterations):
        # Randomly choose 4-6 image filenames from the list
        image_numbers = random.randint(4, 6)
        chosen_filenames = random.sample(image_filenames, image_numbers)

        # Load the chosen images using OpenCV2
        images = [cv2.imread(os.path.join(original_dir_path, filename)) for filename in chosen_filenames]

        # Resize the images to have the same height
        heights = [image.shape[0] for image in images]
        max_height = max(heights)
        resized_images = [cv2.resize(image, (int(image.shape[1] * max_height / image.shape[0]), max_height)) for image in images]

        # Combine the images horizontally using OpenCV2
        combined_image = cv2.hconcat(resized_images)

        # Create the new filename by concatenating the chosen filenames
        new_filename = ''.join(x[0:4] for x in chosen_filenames) + ".jpg"

        # Save the combined image with the new filename
        cv2.imwrite(os.path.join(dir_path, new_filename), combined_image)

        print("Successfully combined and saved the images to", os.path.join(dir_path, new_filename))

# Set the directory path where the images are stored
original_dir_path = "D:/Softwares/Python/CreditCard-OCR/datasets/recognition/train_images"
dir_path = "D:/Softwares/Python/CreditCard-OCR/datasets/recognition/processed"

# Get the list of all image filenames in the directory
image_filenames = [filename for filename in os.listdir(original_dir_path)]

process(11111,original_dir_path,dir_path+'/train')
# process(500,original_dir_path,dir_path+'/val')
# process(500,original_dir_path,dir_path+'/test')
