from io import BytesIO
from typing import Tuple, Any, List

import imageio
import requests
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from skimage import  io, color, morphology, measure, filters, exposure
from sklearn.cluster import DBSCAN
import numpy as np

def download_image(image_url: str) -> np.ndarray:
    """
    Downloads an image from the given URL and converts it into a PIL Image object.

    :param image_url: The URL of the image to download.
    :return: The downloaded image as a PIL Image object.
    """
    try:
        # Send a GET request to fetch the image content
        image = io.imread(image_url)

        return image
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")
        return None


def crop_image(image: np.ndarray, group_centers: list) -> list[ndarray[Any, dtype[floating[_64Bit] | float_]] | ndarray[
    Any, dtype[Any]]] | None:
    """
    Crops the given PIL Image object using the specified boundary.

    :param image: The input image as a PIL Image object.
    :param boundary: A tuple (left, top, right, bottom) specifying the crop box.
    :return: The cropped image as a PIL Image object.
    """
    try:
        image_height, image_width, _ = image.shape  # Assuming img is RGB
        crop_size = 300  # Size of the fixed crop window (300x300 pixels)

        # Calculate half of the crop size
        half_crop = crop_size // 2
        # Iterate through group centers, crop, and save
        cropped_images = []
        for idx, (center_r, center_c) in enumerate(group_centers, start=1):
            # Define the bounding box
            min_row = center_r - half_crop
            max_row = center_r + half_crop
            min_col = center_c - half_crop
            max_col = center_c + half_crop

            # Initialize the cropped image with zeros (black)
            cropped_img = np.zeros((crop_size, crop_size, image.shape[2]), dtype=image.dtype)

            # Calculate overlap with image boundaries
            row_start = max(min_row, 0)
            row_end = min(max_row, image_height)
            col_start = max(min_col, 0)
            col_end = min(max_col, image_width)

            # Calculate corresponding coordinates in the cropped image
            cropped_row_start = row_start - min_row if min_row < 0 else 0
            cropped_row_end = cropped_row_start + (row_end - row_start)
            cropped_col_start = col_start - min_col if min_col < 0 else 0
            cropped_col_end = cropped_col_start + (col_end - col_start)

            # Extract the region from the original image
            cropped_region = image[row_start:row_end, col_start:col_end, :]

            # Place the cropped region into the fixed-size window
            cropped_img[cropped_row_start:cropped_row_end, cropped_col_start:cropped_col_end, :] = cropped_region
            cropped_images.append(cropped_img)
        return cropped_images
    except Exception as e:
        print(f"Failed to crop image: {e}")
        return None

def extract_protein_regions(image: np.ndarray) -> tuple[list[tuple[float, float]], list[Any] | None] | None:
    """
    Extracts the protein regions from the given PIL Image object.
    Args:
        :param image: The input image as a PIL Image object.

    Returns:
        regions: [np.ndarray] Areas where protein regions are extracted.
    """
    try:
        mask = create_protein_mask(image)
        regions = _cluster_regions(mask)

        if regions is None:
            print("No regions found.")
            return None

        # Get image dimensions
        image_height, image_width = image.shape[:2]

        # Calculate percentage positions
        percent_regions = []
        for (center_r, center_c) in regions:
            x_percent = (center_c / image_width) * 100
            y_percent = (center_r / image_height) * 100
            percent_regions.append((x_percent, y_percent))
            print(
                f"Region center (pixels): ({center_r}, {center_c}) -> (percent): ({x_percent:.2f}%, {y_percent:.2f}%)")

        # Crop images based on pixel centers
        cropped_images = crop_image(image, regions)

        return percent_regions, cropped_images
    except Exception as e:
        print(f"Failed to extract protein regions: {e}")
        return None

def create_protein_mask(image: np.ndarray) -> np.ndarray:
    try:
        if image.shape[-1] == 4:
            image = color.rgba2rgb

        # Convert RGB to HEAD color space
        hed = color.rgb2hed(image)

        # Extract the DAB channel
        dab = hed[:, :, 2]

        # Invert the DAB channel
        dab = np.exp(-dab)

        # Normalise the DAB channel to range [0, 1]
        dab = exposure.rescale_intensity(dab, in_range=(0, 1))

        # Apply CLAHE
        dab_clahe = exposure.equalize_adapthist(dab, clip_limit=0.01)

        # Apply Gamma Correction
        gamma = 1
        dab = exposure.adjust_gamma(dab, gamma)

        # Apply a threshold to identify high DAB concentration regions
        threshold_value = filters.threshold_otsu(dab)
        print(f"Otsu's threshold value: {threshold_value}")
        # Adjust threshold if necessary
        high_expr_mask = dab < threshold_value * 0.90

        # Remove small objects and small holes
        high_expr_mask = morphology.remove_small_objects(high_expr_mask, min_size=50)
        high_expr_mask = morphology.remove_small_holes(high_expr_mask, area_threshold=100)
        return high_expr_mask
    except Exception as e:
        print(f"Failed to extract protein channels: {e}")
        return None

def _cluster_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    labeled_mask, num_labels = measure.label(mask, return_num=True, connectivity=2)
    print(f"Total labeled regions: {num_labels}")

    # Step 2: Get properties of labeled regions in a tabular format
    # Include 'centroid' to get the center of each region
    props = measure.regionprops_table(labeled_mask, properties=['label', 'area', 'centroid'])

    labels = np.array(props['label'])
    areas = np.array(props['area'])
    centroids_row = np.array(props['centroid-0'])
    centroids_col = np.array(props['centroid-1'])

    # Define size threshold (e.g., remove regions larger than 500 pixels)
    size_threshold = 500

    # Identify indices of regions smaller than the threshold
    small_indices = np.where(areas < size_threshold)[0]
    small_labels = labels[small_indices]
    small_centroids_row = centroids_row[small_indices]
    small_centroids_col = centroids_col[small_indices]

    # Define clustering parameters
    group_size = 40  # Maximum distance in pixels to consider for grouping


    # Extract centroids of small regions
    small_centroids = np.vstack((small_centroids_row, small_centroids_col)).T  # Shape: (N, 2)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=group_size, min_samples=1).fit(small_centroids)
    cluster_labels = clustering.labels_
    num_clusters = len(set(cluster_labels))
    print(f"Number of clusters formed: {num_clusters}")

    # Calculate group centers as the mean of centroids in each cluster
    group_centers = []
    for cluster in range(num_clusters):
        cluster_points = small_centroids[cluster_labels == cluster]
        center_r, center_c = cluster_points.mean(axis=0)
        group_centers.append((int(round(center_r)), int(round(center_c))))

    return group_centers