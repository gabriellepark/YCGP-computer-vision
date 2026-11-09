# Import functions

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage as ski


# %%

def _plotHist(img, bins =256, range = [0, 256]):
    """
    Function to plot the histogram of an image values, given the image
    bins desired and range determining the x axis ticks.
    This is useful to view histograms pre and post image equalization.
    """
    plt.figure()
    plt.hist(img.flatten(),bins,range, color = 'b')
    plt.xlim([0,256])
    plt.show()


def _padImage(img: np.ndarray, block_size=(16, 16)):
    """"
    Function to pad the image, in case the block dimensions we apply to the image 
    (i.e, mean filtering, block by block equalization) does not divided evenly into the image.
    Given image and block size, the function determines the image size, then calculates the 
    remainder when we divide the image by block_size. It then pads image to the next dimension 
    that enables perfect block divide.
    """
    # Get image current size
    dimX = img.shape[1]
    dimY = img.shape[0]

    # Handle x and y directions separately, in case block size is not square (n x n)
    block_size_x = block_size[1]
    block_size_y = block_size[0]

    # Calculate the remainder, if 0 = no padding, else we will pad to the next evenly divisible dimension.
    remainder_x = dimX%block_size_x
    remainder_y = dimY%block_size_y

    # pad width takes ((top, bottom), (left, right)) - https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    # Since block-reduce (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.block_reduce) 
    # starts at top left, we will pad bottom, right
    if remainder_x != 0:
        pad_amount = block_size_x - remainder_x
        # For ring, wrap works as one edge is the continuation of the other (mimic real eye ring properties)
        img = np.pad(array=img, pad_width=((0, 0), (0, pad_amount)) , mode='wrap')
    
    if remainder_y != 0:
        pad_amount = block_size_y - remainder_y
        # For edges of the ring, use edge as the mode
        img = np.pad(array=img, pad_width= ((0, pad_amount), (0, 0)), mode='edge')

    return img


def _equalizeImg(img, block_shape=(32, 32), show=False):
    """"
    Function to apply histogram equalization on an image, block by block.
    img = image changes are applied to.
    block_shape = dimensions of the block used to apply equalization.
    show = whether to show image histogram or not.
    This set-up replicates the paper's method, where histogram equalization is 
    applied on non-overlapping blocks in the image (Ma et al., Personal Identification Based on Iris Texture Analysis, 3.2.3 Image Enhancement).
    """
    # Copy image to create view as blocks
    img_tmp = img.copy()

    # Use padding again if needed
    img_tmp = _padImage(img_tmp, block_shape)

    # Normalize image to fit the grayscale 0 to 255. Use CV_8U to ensure unint8 values for equalizer.
    img_tmp = cv2.normalize(img_tmp, dst= None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=None)

    # Create a view for block iteration
    blocks = ski.util.view_as_blocks(img_tmp, block_shape)

    # Loop over each block, and apply equalization on that focused area
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
    The function uses OpenCV's .equalizeHist() function on the image, by leveraging
    non-overlapping blocks, and returns enhanhanced image.
    img = image changes are applied to.
    block_size = dimensions of the block used to apply mean filtering.
    show=whether to show image histogram or not.
    Assumes image is already in gray scale.
    """

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

    # May try - https://stackoverflow.com/questions/71587467/what-kind-of-image-processing-techniques-which-i-can-deploy-to-remove-eye-lashes
    # white_image = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
    # cv2.circle(white_image, (inner_circle[0], inner_circle[1]), inner_circle[2]+30, (0, 0, 0), -1)
    # roi = cv2.bitwise_or(median,white_image)
    # #median is the processed binary image

    return img_histEqualization
# %%