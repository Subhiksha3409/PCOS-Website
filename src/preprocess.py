import cv2
import numpy as np
import os

def preprocess_images(image_folder, output_folder):
    """
    Preprocesses images in the specified folder and saves the preprocessed images to the output folder.

    Args:
        image_folder (str): Path to the folder containing images to preprocess.
        output_folder (str): Path to the folder where preprocessed images will be saved.
    """
    # Ensure the input folder exists
    if not os.path.exists(image_folder):
        print(f"Folder {image_folder} does not exist.")
        return

    # Ensure the output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define supported image extensions
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    # Define kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)

    # Iterate through each file in the directory
    for filename in os.listdir(image_folder):
        # Check if the file has a supported image extension
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            image_path = os.path.join(image_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error reading image {image_path}")
                continue

            # Histogram Equalization
            hist_eq = cv2.equalizeHist(image)

            # Median Noise Filter
            median_filtered = cv2.medianBlur(hist_eq, 5)

            # Erosion
            eroded = cv2.erode(median_filtered, kernel, iterations=1)

            # Dilation
            dilated = cv2.dilate(eroded, kernel, iterations=1)

            # Otsu's Thresholding
            _, thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Resize the image to 224x224 pixels
            thresh_resized = cv2.resize(thresh, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Save the preprocessed image to the output directory
            cv2.imwrite(output_path, thresh_resized)
            print(f"Processed and saved {filename} to {output_path}")

def main():
    # Define the path for the test dataset
    input_dirs = [
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/test/PCOS',
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/test/normal'
    ]

    output_dirs = [
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/test/pcos_preprocessed',
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/test/normal_preprocessed'
    ]

    # Process images in the test directory
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        preprocess_images(input_dir, output_dir)

if __name__ == "__main__":
    main()
