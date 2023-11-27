
#-----------------------------------------------------------------------------------------------
#------------------- functions used in the predicare_part_1 notebook ----------------------------
#-----------------------------------------------------------------------------------------------

def read_data(data_dir):
  # Initialize lists for images, labels
  images = []
  labels = []
  classes = {'normal': 0, 'benign': 1, 'malignant' : 2}

  for class_name, class_label in classes.items():
      class_dir = os.path.join(data_dir, class_name)
      for image_file in os.listdir(class_dir):
          if image_file.endswith(".png") and 'mask' not in image_file:
              # Load the original image
              image_path = os.path.join(class_dir, image_file)
              images.append(image_path)

              class_label = classes[class_name]
              labels.append(class_label)

  return images, labels


def count_images_in_specific_folders(root_directory):
  # List of folders to look for
  target_folders = ['normal', 'benign', 'malignant']

  # Initialize a list to store results for each target folder
  folder_counts = []

  # Iterate through the target folders
  for folder in target_folders:
    folder_path = os.path.join(root_directory, folder)

    # Skip non-existent folders
    if not os.path.exists(folder_path):
      continue

    # Initialize counters for the current folder
    total_images = 0
    images_without_mask = 0

    # Iterate through files in the current folder
    for file in os.listdir(folder_path):
        # Check if the file is a PNG image
        if file.lower().endswith('.png'):
          total_images += 1

          # Check if the file name contains 'mask'
          if 'mask' not in file.lower():
            images_without_mask += 1

    # Store the counts for the current folder
    folder_counts.append({
      'folder_name': folder,
      'total_images': total_images,
      'images_without_mask': images_without_mask
    })

  return folder_counts


def visualize_counts_bar_chart(results):
  labels = [result['folder_name'] for result in results]
  total_images = [result['total_images'] for result in results]
  images_without_mask = [result['images_without_mask'] for result in results]

  # Bar chart
  bar_width = 0.35
  index = range(len(labels))
  fig, ax = plt.subplots()

  # Bar chart for total images
  bars_total = ax.bar(index, total_images, bar_width, label='Total (Images and Masks)', color='#DF7E10')

  # Bar chart for images without 'mask'
  bars_without_mask = ax.bar([i + bar_width for i in index], images_without_mask, bar_width, label='Only Images', color='#FFB38A')

  # Display numbers on top of each bar
  for bar, value in zip(bars_total, total_images):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value}', ha='center', va='bottom')

  for bar, value in zip(bars_without_mask, images_without_mask):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value}', ha='center', va='bottom')

  ax.set_xlabel('Folder')
  ax.set_ylabel('Number of png-Files')
  ax.set_title('Number of png-Files in Folders')
  ax.set_xticks([i + bar_width / 2 for i in index])
  ax.set_xticklabels(labels)
  ax.legend()

  plt.show()


def process_masks(input_folder, output_folder):
  os.makedirs(output_folder, exist_ok=True)

  # Iterate through the files in the input folder
  for filename in os.listdir(input_folder):
    if filename.endswith(".png") and 'mask' not in filename:
      # Extract the image name without the ".png" extension
      image_name = os.path.splitext(filename)[0]
      mask1_path = os.path.join(input_folder, f"{image_name}_mask.png")
      mask2_path = os.path.join(input_folder, f"{image_name}_mask_1.png")
      mask3_path = os.path.join(input_folder, f"{image_name}_mask_2.png")

      # Check if all three masks exist
      if os.path.exists(mask1_path) and os.path.exists(mask2_path) and os.path.exists(mask3_path):
        # Copy original image into the output folder
        original_image_path = os.path.join(input_folder, f"{image_name}.png")
        shutil.copy(original_image_path, output_folder)

        # Load the masks
        mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
        mask3 = cv2.imread(mask3_path, cv2.IMREAD_GRAYSCALE)

        # Merge the masks
        merged_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)  # merging three masks

        # Save the merged mask in the output folder
        merged_mask_path = os.path.join(output_folder, f"{image_name}_mask.png")
        cv2.imwrite(merged_mask_path, merged_mask)

        print(f"Merged three masks for {image_name}")

      # Check if two masks exist
      elif os.path.exists(mask1_path) and os.path.exists(mask2_path):
        # Copy original image into the output folder
        original_image_path = os.path.join(input_folder, f"{image_name}.png")
        shutil.copy(original_image_path, output_folder)

        # Load the masks
        mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

        # Merge the masks
        merged_mask = cv2.bitwise_or(mask1, mask2)  # merging two masks

        # Save the merged mask in the output folder
        merged_mask_path = os.path.join(output_folder, f"{image_name}_mask.png")
        cv2.imwrite(merged_mask_path, merged_mask)

        print(f"Merged two masks for {image_name}")

      else:
        # If only one mask exists, copy the original image and the mask to the output folder
        original_image_path = os.path.join(input_folder, f"{image_name}.png")
        shutil.copy(original_image_path, output_folder)

        if os.path.exists(mask1_path):
          original_mask_path = os.path.join(output_folder, f"{image_name}_mask.png")
          shutil.copy(mask1_path, original_mask_path)

        print(f"Copied original image for {image_name}")


def rotate_images(input_directory, output_directory, times=1, rotation_angle=40):
  os.makedirs(output_directory, exist_ok=True)

  image_files = [f for f in os.listdir(input_directory) if f.endswith('.png') and 'mask' not in f]

  for image_file in image_files[:times]:
    image_name = os.path.splitext(image_file)[0]
    image_path = os.path.join(input_directory, image_file)
    original_image = cv2.imread(image_path)

    # Get the image dimensions
    height, width = original_image.shape[:2]

    # Create a rotation matrix for the desired angle
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1)

    # Apply the rotation to the original image with the nearest-neighbor fill mode
    rotated_image = cv2.warpAffine(original_image, rotation_matrix, (width, height), (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Determine the name of the rotated image
    rotated_image_name = f"{image_file.replace('.png', '')}_rotation.png"
    rotated_image_path = os.path.join(output_directory, rotated_image_name)

    # Save the rotated image
    cv2.imwrite(rotated_image_path, rotated_image)

    # Copy the original image to the output folder
    original_image_output_path = os.path.join(output_directory, image_file)
    shutil.copy(image_path, original_image_output_path)

    # Apply rotation to the mask
    mask_name = f"{image_name}_mask.png"
    mask_path = os.path.join(input_directory, mask_name)
    original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Apply rotation to the mask
    rotated_mask = cv2.warpAffine(original_mask, rotation_matrix, (width, height), (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Determine the name of the rotated mask
    rotated_mask_name = f"{mask_name.replace('_mask.png', '')}_rotation_mask.png"
    rotated_mask_path = os.path.join(output_directory, rotated_mask_name)

    # Save the rotated mask
    cv2.imwrite(rotated_mask_path, rotated_mask)

    # Copy the original mask to the output folder
    original_mask_output_path = os.path.join(output_directory, mask_name)
    shutil.copy(mask_path, original_mask_output_path)


def shear_images(input_directory, output_directory, times=1, shear_x=0.2, shear_y=0.2):
  os.makedirs(output_directory, exist_ok=True)

  image_files = [f for f in os.listdir(input_directory) if f.endswith('.png') and 'mask' not in f]

  for image_file in image_files[:times]:
    image_name = os.path.splitext(image_file)[0]
    image_path = os.path.join(input_directory, image_file)
    original_image = cv2.imread(image_path)

    # Get the image dimensions
    height, width = original_image.shape[:2]

    # Define shear parameters
    shear_matrix = np.float32([[1, shear_x, 0],
                                [shear_y, 1, 0]])

    # Apply shear to the original image
    shear_image = cv2.warpAffine(original_image, shear_matrix, (width, height), (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Determine the name of the sheared image
    shear_image_name = f"{image_file.replace('.png', '')}_shear.png"
    shear_image_path = os.path.join(output_directory, shear_image_name)

    # Save the sheared image
    cv2.imwrite(shear_image_path, shear_image)

    # Copy the original image to the output folder
    original_image_output_path = os.path.join(output_directory, image_file)
    shutil.copy(image_path, original_image_output_path)

    # Apply shear to the mask
    mask_name = f"{image_name}_mask.png"
    mask_path = os.path.join(input_directory, mask_name)
    original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Apply shear to the mask
    shear_mask = cv2.warpAffine(original_mask, shear_matrix, (width, height), (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Determine the name of the sheared mask
    shear_mask_name = f"{mask_name.replace('_mask.png', '')}_shear_mask.png"
    shear_mask_path = os.path.join(output_directory, shear_mask_name)

    # Save the sheared mask
    cv2.imwrite(shear_mask_path, shear_mask)

    # Copy the original mask to the output folder
    original_mask_output_path = os.path.join(output_directory, mask_name)
    shutil.copy(mask_path, original_mask_output_path)


def zoom_images(input_directory, output_directory, times=1, zoom_factor=1.2, output_size=(512, 512)):
  os.makedirs(output_directory, exist_ok=True)

  image_files = [f for f in os.listdir(input_directory) if f.endswith('.png') and 'mask' not in f]

  for image_file in image_files[:times]:
    image_name = os.path.splitext(image_file)[0]
    image_path = os.path.join(input_directory, image_file)
    original_image = cv2.imread(image_path)

    # Get the image dimensions
    height, width = original_image.shape[:2]

    # Define zoom parameters
    zoom_center_x, zoom_center_y = width // 2, height // 2

    # Apply zoom to the original image
    zoom_image = original_image[
        int(zoom_center_y - (output_size[1] / (2 * zoom_factor))):int(zoom_center_y + (output_size[1] / (2 * zoom_factor))),
        int(zoom_center_x - (output_size[0] / (2 * zoom_factor))):int(zoom_center_x + (output_size[0] / (2 * zoom_factor)))
    ]

    # Determine the name of the zoomed image
    zoom_image_name = f"{image_file.replace('.png', '')}_zoom.png"
    zoom_image_path = os.path.join(output_directory, zoom_image_name)

    # Save the zoomed image
    cv2.imwrite(zoom_image_path, zoom_image)

    # Copy the original image to the output folder
    original_image_output_path = os.path.join(output_directory, image_file)
    shutil.copy(image_path, original_image_output_path)

    # Apply zoom to the mask
    mask_name = f"{image_name}_mask.png"
    mask_path = os.path.join(input_directory, mask_name)
    original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Apply zoom to the mask
    zoom_mask = original_mask[
        int(zoom_center_y - (output_size[1] / (2 * zoom_factor))):int(zoom_center_y + (output_size[1] / (2 * zoom_factor))),
        int(zoom_center_x - (output_size[0] / (2 * zoom_factor))):int(zoom_center_x + (output_size[0] / (2 * zoom_factor)))
    ]

    # Determine the name of the zoomed mask
    zoom_mask_name = f"{mask_name.replace('_mask.png', '')}_zoom_mask.png"
    zoom_mask_path = os.path.join(output_directory, zoom_mask_name)

    # Save the zoomed mask
    cv2.imwrite(zoom_mask_path, zoom_mask)

    # Copy the original mask to the output folder
    original_mask_output_path = os.path.join(output_directory, mask_name)
    shutil.copy(mask_path, original_mask_output_path)


def translate_images(input_directory, output_directory, times=1, translation_x=0.2, translation_y=0.2, output_size=(512, 512)):
  os.makedirs(output_directory, exist_ok=True)

  image_files = [f for f in os.listdir(input_directory) if f.endswith('.png') and 'mask' not in f]

  for image_file in image_files[:times]:
    image_name = os.path.splitext(image_file)[0]
    image_path = os.path.join(input_directory, image_file)
    original_image = cv2.imread(image_path)

    # Get the image dimensions
    height, width = original_image.shape[:2]

    # Define translation parameters
    translation_matrix = np.float32([[1, 0, translation_x * width],
                                    [0, 1, translation_y * height]])

    # Apply translation to the original image
    translation_image = cv2.warpAffine(original_image, translation_matrix, output_size, borderMode=cv2.BORDER_REPLICATE)

    # Determine the name of the translated image
    translation_image_name = f"{image_file.replace('.png', '')}_translation.png"
    translation_image_path = os.path.join(output_directory, translation_image_name)

    # Save the translated image
    cv2.imwrite(translation_image_path, translation_image)

    # Copy the original image to the output folder
    original_image_output_path = os.path.join(output_directory, image_file)
    shutil.copy(image_path, original_image_output_path)

    # Apply translation to the mask
    mask_name = f"{image_name}_mask.png"
    mask_path = os.path.join(input_directory, mask_name)
    original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Apply translation to the mask
    translation_mask = cv2.warpAffine(original_mask, translation_matrix, output_size, borderMode=cv2.BORDER_REPLICATE)

    # Determine the name of the translated mask
    translation_mask_name = f"{mask_name.replace('_mask.png', '')}_translation_mask.png"
    translation_mask_path = os.path.join(output_directory, translation_mask_name)

    # Save the translated mask
    cv2.imwrite(translation_mask_path, translation_mask)

    # Copy the original mask to the output folder
    original_mask_output_path = os.path.join(output_directory, mask_name)
    shutil.copy(mask_path, original_mask_output_path)


def copy_files(input_directory, output_directory):
  os.makedirs(output_directory, exist_ok=True)

  image_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

  for image_file in image_files:
    image_path = os.path.join(input_directory, image_file)
    shutil.copy(image_path, os.path.join(output_directory, image_file))



#-----------------------------------------------------------------------------------------------
#------------------- functions used in the predicare_part_2 notebook ----------------------------
#-----------------------------------------------------------------------------------------------

# Read balanced data set
def read_balanced_data(data_dir):
  ''' The function reads the images and corresponding masks stored according their classes
  normal, benign, and malignant. The data directory is the one, where the three
  folders are stored.'''

  # Initialize lists for images, mask, labels
  images = []
  masks = []
  labels = []
  classes = {'normal': 0, 'benign': 1, 'malignant' : 2}

  for class_name, class_label in classes.items():
      class_dir = os.path.join(data_dir, class_name)
      for image_file in os.listdir(class_dir):
          if image_file.endswith(".png") and 'mask' not in image_file:
              # Load the original image
              image_path = os.path.join(class_dir, image_file)
              images.append(image_path)

              # Find all masks associated with the image
              image_name = os.path.splitext(image_file)[0]# Remove the file extension
              matching_masks = [f for f in os.listdir(class_dir) if f == image_name + '_mask.png']
              for mask_file in matching_masks:
                mask_path = os.path.join(class_dir, mask_file)
                masks.append(mask_path)

              class_label = classes[class_name]
              labels.append(class_label)

  return images, masks, labels


# Function to apply Local Binary Pattern (LBP) algorithm on loaded images
def compute_lbps(images):
  ''' The functions applies the local binary pattern (LBP) algorithm'''
  lbps = []
  for image_path in images:
    # Load the original image
    image = io.imread(image_path)

    # Compute LBP for the image
    lbp_image = feature.local_binary_pattern(color.rgb2gray(image), P=8, R=1, method='uniform')
    lbp_image = np.expand_dims(lbp_image, axis = -1)
    lbps.append(lbp_image)

  return lbps


def compute_hogs(images):
  ''' The function applies the histogram of orientated gradient (HOG) algorithm'''
  hogs = []
  for image_path in images:
    # Load the original image
    image = io.imread(image_path)
    # Compute HOG for the image
    fd, hog_image = feature.hog(
        image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1,
    )
    # Optionally, rescale intensity for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Append the HOG features to the list
    hogs.append(hog_image_rescaled)

  return hogs


def visualize_images(images, masks, lbps, hogs, labels):
  num_samples = len(images)

  for i in range(num_samples):
    # Load original image
    image = io.imread(images[i])

    # Load mask
    mask = io.imread(masks[i])

    # Display original image, mask, LBP-image, and HOG-image side by side
    plt.figure(figsize=(18, 4))

    plt.subplot(1, 5, 1)
    plt.imshow(image)
    plt.title(f'Original Image\nLabel: {labels[i]}')

    plt.subplot(1, 5, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth Mask')

    plt.subplot(1, 5, 3)
    plt.imshow(lbps[i], cmap='gray')
    plt.title('Local Binary Pattern (LBP)')

    plt.subplot(1, 5, 4)
    # Compute HOG features and visualize
    plt.imshow(hogs[i], cmap='gray')
    plt.title('Histogram of Oriented Gradients (HOG)')

    plt.show()


def preprocess_images_unet(data_set):
  ''' The function modifies the data set given as a pandaframe and separates
  the origin images, LBP-images, HOG-images, masks, and labels into numpy arrays.
  This preprocessing is mandatory to apply the U-net.'''

  height, width, channels = 128, 128, 3  # Assuming RGB images as input

  # Initialize empty arrays
  X_image = np.zeros((len(data_set), height, width, channels), dtype=np.uint8)
  X_lbp = np.zeros((len(data_set), height, width, 1), dtype=np.uint8)
  X_hog = np.zeros((len(data_set), height, width, 1), dtype=np.uint8)
  masks = np.zeros((len(data_set), height, width, 1), dtype=np.bool)
  labels = np.zeros(len(data_set), dtype=np.uint8)

  # Iterate through the DataFrame to populate the initalized arrays
  for idx, row in data_set.iterrows():
    image_path = row['images']
    lbp = row['lbps']
    hog = row['hogs']
    mask_path = row['masks']
    label = row['labels']

    # Load and resize the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))

    # Load and resize the LBP image
    lbp_resized = cv2.resize(lbp, (width, height))
    # Add a singleton dimension to make it (height, width, 1)
    lbp_resized = np.expand_dims(lbp_resized, axis=-1)

    # Load and resize the HOG image
    hog_resized = cv2.resize(hog, (width, height))
    # Add a singleton dimension to make it (height, width, 1)
    hog_resized = np.expand_dims(hog_resized, axis=-1)

    # Load and resize the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height))
    mask = np.expand_dims(mask, axis=-1)


    X_image[idx] = image
    X_lbp[idx] = lbp_resized
    X_hog[idx] = hog_resized
    masks[idx] = mask
    labels[idx] = label

  return X_image, X_lbp, X_hog, masks, labels


def visualize_images_nparray(images, lbps, hogs, masks, labels, num_samples=3):
  random_indices = np.random.randint(0, len(images), num_samples)

  for i in random_indices:
    sample_image = images[i]
    sample_lbp = lbps[i]
    sample_hog = hogs[i]
    true_mask = masks[i, :, :, 0]
    label = labels[i]

    # Plot the images and masks
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(sample_image)
    plt.title(f'Image - Label: {label}')

    plt.subplot(1, 4, 2)
    plt.imshow(sample_lbp[:, :, 0], cmap='gray')
    plt.title('LBP Image')

    plt.subplot(1, 4, 3)
    plt.imshow(sample_hog[:, :, 0], cmap='gray')
    plt.title('HOG Image')

    plt.subplot(1, 4, 4)
    plt.imshow(true_mask, cmap='gray')
    plt.title('Ground Truth Mask')

    plt.show()


def visualize_results_unet(images, true_masks, predicted_masks, labels, num_samples=5):
  for i in range(num_samples):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f'Test Image\nLabel: {labels[i]}')

    plt.subplot(1, 3, 2)
    plt.imshow(true_masks[i].squeeze(), cmap='gray')
    plt.title('Ground Truth Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_masks[i].squeeze(), cmap='gray')
    plt.title('Predicted Mask')

    plt.show()


def resize_images_and_masks(images, lbp_images, hog_images, masks, new_size=(197, 197)):
   # Calculate zoom factors
  zoom_factors = (new_size[0] / images.shape[1], new_size[1] / images.shape[2])

  # Resize images
  resized_images = np.array([ndimage.zoom(img, zoom_factors + (1,), order=1) for img in images])

  # Resize LBP and HOG images
  resized_lbp_images = np.array([ndimage.zoom(img, zoom_factors + (1,), order=1) for img in lbp_images])
  resized_hog_images = np.array([ndimage.zoom(img, zoom_factors + (1,), order=1) for img in hog_images])

  # Resize masks
  resized_masks = np.array([ndimage.zoom(mask, zoom_factors + (1,), order=1) for mask in masks])

  return resized_images, resized_lbp_images, resized_hog_images, resized_masks