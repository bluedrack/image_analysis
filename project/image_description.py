# Check is at least python 3.9
import sys 
#assert (sys.version_info.major == 3) and (sys.version_info.minor == 9)
# Other global libraries
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import wget
import os
from typing import Callable

def display_samples(images: np.ndarray, labels:np.ndarray, title: str, cnt: list = None):
    """
    Display images along with labels. 
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images
    labels: np.ndarray (N)
        List of labels associated with the input image
    title: str
        Title of the plot
    cnt: list
        List of contours to display (only used for exercise 1.3 and more)
    """

    # Get the number of images, columns, and rows
    n = len(images)
    n_cols = 8
    r_rows = np.ceil(n/n_cols).astype(int)
    
    # Define plot
    _, axes = plt.subplots(r_rows, n_cols, figsize=(14, 2*r_rows))
    axes = axes.ravel()
    
    
    # Plot all images and labels
    for i in range(n):
        axes[i].imshow(images[i], interpolation="nearest")
        axes[i].axis("off")
        axes[i].set_title(labels[i])
        # Check if need to display contour
        if cnt is not None and len(cnt) == n:
            axes[i].plot(cnt[i][:, 0], cnt[i][:, 1], 'r-*')

    # Set title
    plt.suptitle(title)
    plt.tight_layout()
        

def extract_label(images: np.ndarray, labels: np.ndarray, target_label: int):
    """
    The function returns only the images that have target_label as labels.
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images - handwritten digits 
    labels: np.ndarray (N)
        List of labels associated with the input image
    target_label: int
        Selected target label

    Return
    ------
    img_extract: np.ndarray (M, 28, 28)
        Extracted images that have target_label as label (M should be lower than N).
    """

    n, d, _ = np.shape(images) 
    img_extract = np.zeros((30, d, d))
    
    # ------------------
    # Your code here ... 
    # ------------------
    img_extract = images[labels == target_label]
    return img_extract

from skimage.morphology import *
def preprocess(images: np.ndarray):
    """
    Apply the processing step to images to achieve better data uniformity.
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images

    Return
    ------
    img_process: np.ndarray (N, 28, 28)
        Processed images.
    """

    # Get the shape of input data and set dummy values
    n, d, _ = np.shape(images) 
    img_process = np.zeros_like(images)
    
    # ------------------
    # Your code here ... 
    # ------------------
    footprint = disk(1)
    threshold = 100
    for i in range(n):
        img = images[i]
        img = img > threshold
        img = closing(img, footprint=footprint)
        img = remove_small_holes(img, area_threshold=200)
        img_process[i] = img
    return img_process

import skimage.measure
def find_contour(images: np.ndarray):
    """
    Find the contours for the set of images
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images to process

    Return
    ------
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    """

    # Get number of images to process
    N, _, _ = np.shape(images)
    # Fill in dummy values (fake points)
    contours = [np.array([[0, 0], [1, 1]]) for i in range(N)]

    # ------------------
    # Your code here ... 
    # ------------------
    for i in range(N):
        img = images[i]
        contours[i] = skimage.measure.find_contours(img)[0]
        contours[i] = np.flip(contours[i], axis=1)
    return contours
    
import numpy.fft as fft
def compute_descriptor_padding(contours: np.ndarray, n_samples: int = 11):
    """
    Compute Fourier descriptors of input images
    
    Args
    ----
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    n_samples: int
        Number of samples to consider. If the contour length is higher, discard the remaining part. If it is shorter, add padding.
        Make sure that the first element of the descriptor represents the continuous component.

    Return
    ------
    descriptors: np.ndarray complex (N, n_samples)
        Computed complex Fourier descriptors for the given input images
    """

    N = len(contours)
    # Look for the number of contours
    descriptors = np.zeros((N, n_samples), dtype=np.complex_)

    # ------------------
    # Your code here ... 
    # ------------------

    for i in range(N):
        contour = contours[i]
        n = len(contour)
        if n < n_samples:
            contour = np.concatenate((contour, np.zeros((n_samples - n, 2))))
        elif n > n_samples:
            contour = contour[:n_samples]
        descriptors[i] = fft.fft(contour[:, 0] + 1j * contour[:, 1])


    return descriptors

def plot_features(features_a: np.ndarray, features_b: np.ndarray, label_a: str, label_b: str, title: str):
    """
    Plot feature components a and b.
    
    Args
    ----
    features_a: np.ndarray (N, D)
        Feature a with N samples and D complex features. 
    features_b: np.ndarray (N, D)
        Feature b with N samples and D complex features.
    label_a: str
        Name of the feature a.
    label_b: str
        Name of the feature b.
    """

    # Number of paris to display
    n_features = features_a.shape[1]
    # Define pairs for 2D plots
    pairs = np.array(range(2*np.ceil(n_features / 2).astype(int)))
    # Check if odd lenght, shift second feature to have pairs
    if n_features % 2 == 1:
        pairs[2:] = pairs[1:-1]
    # Convert to 2d array
    pairs = pairs.reshape(-1, 2)

    # Plot each pairs and labels
    n_plots = len(pairs)
    _, axes = plt.subplots(3, n_plots, figsize=(15, 8))
    
    for i, (pa, pb) in enumerate(pairs):
        # Real
        axes[0, i].scatter(np.real(features_a[:, pa]), np.real(features_a[:, pb]), label=label_a, s=10, alpha=0.1)
        axes[0, i].scatter(np.real(features_b[:, pa]), np.real(features_b[:, pb]), label=label_b, s=10, alpha=0.1)
        axes[0, i].set_xlabel("Component {}".format(pa))
        axes[0, i].set_ylabel("Component {}".format(pb))
        axes[0, i].set_title("Real {} vs {}".format(pa, pb))
        axes[0, i].legend()
        # Imag
        axes[1, i].scatter(np.imag(features_a[:, pa]), np.imag(features_a[:, pb]), label=label_a, s=10, alpha=0.1)
        axes[1, i].scatter(np.imag(features_b[:, pa]), np.imag(features_b[:, pb]), label=label_b, s=10, alpha=0.1)
        axes[1, i].set_xlabel("Component {}".format(pa))
        axes[1, i].set_ylabel("Component {}".format(pb))
        axes[1, i].set_title("Imag. {} vs {}".format(pa, pb))
        axes[1, i].legend()
        # Abs
        axes[2, i].scatter(np.abs(features_a[:, pa]), np.abs(features_a[:, pb]), label=label_a, s=10, alpha=0.1)
        axes[2, i].scatter(np.abs(features_b[:, pa]), np.abs(features_b[:, pb]), label=label_b, s=10, alpha=0.1)
        axes[2, i].set_xlabel("Component {}".format(pa))
        axes[2, i].set_ylabel("Component {}".format(pb))
        axes[2, i].set_title("Abs. {} vs {}".format(pa, pb))
        axes[2, i].legend()

    plt.suptitle(title)
    plt.tight_layout()


def linear_interpolation(contours: np.ndarray, n_samples: int = 11):
    """
    Perform interpolation/resampling of the contour across n_samples.
    
    Args
    ----
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    n_samples: int
        Number of samples to consider along the contour.

    Return
    ------
    contours_inter: np.ndarray complex (N, n_samples, 2)
        Interpolated contour with n_samples
    """

    N = len(contours)
    contours_inter = np.zeros((N, n_samples, 2))
    
    # ------------------
    # Your code here ... 
    # ------------------
    #print("n_smaples: ", n_samples)
    #print("contour: ", contours)
    for i in range(N):
        contour = contours[i]
        x, y = zip(*contour)
    
        # Interpolate x values
        new_x = np.interp(np.linspace(0, 1, n_samples), np.linspace(0, 1, len(x)), x)
    
        # Interpolate y values
        new_y = np.interp(np.linspace(0, 1, n_samples), np.linspace(0, 1, len(y)), y)
    
        # Combine new x and y values into a list of (x, y) coordinates
        interpolated_coords = [(x_val, y_val) for x_val, y_val in zip(new_x, new_y)]
        contours_inter[i] = interpolated_coords

    return contours_inter


def compute_reverse_descriptor(descriptor: np.ndarray, n_samples: int = 11):
    """
    Reverse a Fourier descriptor to xy coordinates given a number of samples.
    
    Args
    ----
    descriptor: np.ndarray (D,)
        Complex descriptor of length D.
    n_samples: int
        Number of samples to consider to reverse transformation.

    Return
    ------
    x: np.ndarray complex (n_samples,)
        x coordinates of the contour
    y: np.ndarray complex (n_samples,)
        y coordinates of the contour
    """

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    
    # ------------------
    # Your code here ... 
    # ------------------
    a  = np.fft.ifft(descriptor)
    x = a.real
    y = a.imag
    return x, y


import skimage.measure
def find_contour(images: np.ndarray):
    """
    Find the contours for the set of images
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images to process

    Return
    ------
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    """

    # Get number of images to process
    N, _, _ = np.shape(images)
    # Fill in dummy values (fake points)
    contours = [np.array([[0, 0], [1, 1]]) for i in range(N)]

    # ------------------
    # Your code here ... 
    # ------------------
    for i in range(N):
        img = images[i]
        contours[i] = skimage.measure.find_contours(img)[0]
        contours[i] = np.flip(contours[i], axis=1)
    return contours
    