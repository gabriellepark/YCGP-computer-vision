# Import functions

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage as ski


# %%

def _plotHist(img, bins =256, range = [0, 256]):
    plt.figure()
    plt.hist(img.flatten(),bins,range, color = 'b')
    plt.xlim([0,256])
    plt.show()


def _padImage(img: np.ndarray, block_size=(16, 16)):
    # Get image current size
    dimX = img.shape[1]
    dimY = img.shape[0]

    # Handle x and y directions separately, in case block size is not square (n x n)
    block_size_x = block_size[1]
    block_size_y = block_size[0]

    remainder_x = dimX%block_size_x
    remainder_y = dimY%block_size_y

    # pad width takes ((top, bottom), (left, right))
    # Since block-reduce starts at top left, we will pad bottom, right
    if remainder_x != 0:
        pad_amount = block_size_x - remainder_x
        # For ring, wrap works as one edge is the continuation of the other
        img = np.pad(array=img, pad_width=((0, 0), (0, pad_amount)) , mode='wrap')
    
    if remainder_y != 0:
        pad_amount = block_size_y - remainder_y
        # For edges of the ring, use edge as the mode
        img = np.pad(array=img, pad_width= ((0, pad_amount), (0, 0)), mode='edge')

    return img


def _equalizeImg(img, block_shape=(32, 32), show=False):
    # Copy image to create view as blocks
    img_tmp = img.copy()

    # Use padding again if needed
    img_tmp = _padImage(img_tmp, block_shape)

    # Normalize image to fit the grayscale 0 to 255. Use CV_8U to ensure unint8 values for equalizer.
    img_tmp = cv2.normalize(img_tmp, dst= None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=None)

    # Create a view
    blocks = ski.util.view_as_blocks(img_tmp, block_shape)

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            # Equalize block
            blocks[i, j] = cv2.equalizeHist(blocks[i, j])

    # View equalized histogram
    if show:
        _plotHist(img_tmp)

    return img_tmp


def enhanceImage(img, block_size=(16, 16), show=False):
    """
    Function to enhance the quality of the normalized image.
    The function uses OpenCV's .equalizeHist() function on the image and returns
    enhanhanced image.
    Assumes image is already in gray scale.
    """

    # plt.imshow(img, cmap='gray')
    # Save original image size
    og_img_size = img.shape

    # Plot current image histogram
    if show:
        _plotHist(img, bins=256, range=[0, 256])

    # The paper uses "mean of each 16x16 small block" to estimate background illumination
    # (Ma et al., Personal Identification Based on Iris Texture Analysis, 3.2.3 Image Enhancement)

    # Create a 16 x 16 mean filter to do the same
    # Chosen normalized image following paper is width = 512 and height = 64
    # 512/16 = 32, 64/16 = 4, so final image size after mean pooling should be 32 by 4

    # First, call padding function in case image does not fit block size perfectly
    img_padded = _padImage(img, block_size=block_size)

    # Use a resize function to get final frame with mean as the function
    background_illumination = ski.measure.block_reduce(img_padded, block_size=block_size, func=np.mean)

    # Recover normalized size with bicubic interpolation
    # Using [::-1] to reverse original size so it matches (width, height) convention of the .resize() function
    background_illumination_normalized = cv2.resize(background_illumination, og_img_size[::-1],
                                                    interpolation=cv2.INTER_CUBIC)

    # Remove background illumination from normalized image
    enhanced_image = cv2.subtract(img, background_illumination_normalized)

    # Use histogram equalization to enhance image quality for dark settings
    img_histEqualization = _equalizeImg(enhanced_image, show=show)

    return img_histEqualization
# %%