# FeatureExtraction.py: filtering the iris and extracting features;


# Import functions
import cv2
import numpy as np



# Implementation based on Ma et al. 2003
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1251145&utm_source=scopus&getft_integrator=scopus


def spatial_filter(dx, dy, size=27):
    """
    Function to get spatial filter, based on Ma's implementation.
    dx = space constant of Gaussian envelope along x direction
    dy = space constant of Gaussian envelope along y direction
    size = width and height of the square kernel in pixels
    return: size x size kernel (Gaussian envelope x circularly symmetric sinusoid), shifted to zero mean to be convolved with ROI using cv2.filter2D
    """
    # Spatial filter: Gaussian envelope x modulating function
    # Ma's defined filter uses a circularly symmetric sinusoid (M1),
    # unlike the even-symmetric Gabor filter's oriented sinusoid
    # derived from Ma's equation #3
    f = 1 / dx  # frequency as one cosine cycle across gaussian's width
    half = size // 2  # size determined to cover about +/- 3*dx for wider channels

    # Intialize kernel
    kernel = np.zeros((size, size), dtype=np.float32)
    # Iterates through each cell in the grid
    for i in range(size):
        for j in range(size):
            # Pixel (i, j) is coordinates (x, y) centered at (0, 0)
            # Convert array indices to coordinates centered on kernel middle
            x = j - half
            y = i - half

            # Equation from Ma's implementation
            m1 = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))

            gauss = np.exp(-0.5 * (x**2 / dx**2 + y**2 / dy**2)) / (2 * np.pi * dx * dy)
            kernel[i, j] = gauss * m1

    # Subtract average kernel weight so all weights sum to 0
    kernel = kernel - kernel.mean()
    return kernel



def extract_features(enhanced):
    """
    Function to extract features from image
    Enhanced: enhanced image from ImageEnhancement.py
    Returns: 1D feature vector of length 1536

    Filters ROI with two spatial filter channels then summarizes
    each 8 x 8 block of the filtered images with its mean and average absolute deviation


    """

    # Extract upper portion of the normalized image closer to the pupil
    # Ma's Section 3.3.2 identifies ROI as 48 x 512
    # Convert to float32 so filtering can produce negative values
    roi = enhanced[:48, :].astype(np.float32)

    # Initialize feature list
    features = []

    # Section 3.3.1 details the first and second channel
    # first channel: dx = 3, dy = 1.5
    # second channel: dx = 4.5, dy = 1.5
    # Each channel captures texture at a different scale/frequency so we iterate through both
    for dx, dy in [(3, 1.5), (4.5, 1.5)]:
        # Equation 4: Fi(x, y) = integral of I(x1, y1) * Gi(x - x1, y - y1)
        # cv2.filter2D slides the spatial filter over every pixel of the ROI and computes weighted sum
        # Filter is symmetric so filter2D's correlation equals the convolution in Eqn 4
        # Eqn 5 uses the magnitude |Fi| so take the absolute value
        filtered = np.abs(cv2.filter2D(roi, cv2.CV_32F, spatial_filter(dx, dy)))

        # Extract statistical features in each 8 x 8 block of two filtered images
        # Total number of small blocks: (48 x 512) / (8 x 8) = 384 blocks per channel, 768 in total
        # Iterate through the filtered image in 8 x 8 blocks
        for i in range(0, 48, 8):
            for j in range(0, 512, 8):
                # Get each 8x8 block
                block = filtered[i:i + 8, j:j + 8]
                m = block.mean() # Eqn 5: mean of magnitude of filtered block
                sigma = np.abs(block - m).mean() # Eqn 5: average absolute deviation
                # Add to feature list
                features += [m, sigma]

    # Eqn 6: V = [m1, sigma1, m2, sigma2, ..., m768, sigma768]^T
    return np.array(features)







# ___________________________________________________________________________________________________

# Notes from Ma et al. 2003:
# frequency info should account for major differences of irises from different people
#  capture discriminating freq info which reflects local structure of iris
# majority of useful info is in freq band of about three octaves
# bank of filters is constructed to reliably acquire such info in spatial domain
# coeff of filtered image effectively indicate the freq distribution of image
# two stat values extracted from each small region in filtered image to represent local texture info of iris

# feature vector is ordered collection of all features of local regions

# spatial filters
# circularly symmetric sinusoidal function
# even symmetric gabor filters
# modulating function of defined filter and gabor filter
# upper portion of normalized iris image closer to pupil provide the most useful texture info
# ROI
# delta x is 3, delta y is 1.5 in first channel
# 4.5 and 1.5 in second

# feature vector
# double integral of roi * ith channel of spatial filters
# extract statistical features in each 8x8 small block of two filtered images
# generates 1536 feature components
# mean and average absolute deviation of magnitude of each filtered block
# arranged to form a 1d feature vector





# __________________________________________________________________________________________________________


# Notes from Daugman's. Not used.
# https://www.robots.ox.ac.uk/~az/lectures/est/iris.pdf
# normalized iris projected onto complex valued 2-d gabor wavelets
# phase information extracted for encoding the iris pattern
# 2048 phase bits computed for each iris
# patch wise phase quantization of iris pattern


# local regions of an iris are projected onto quadrature 2d gabor wavelets, generating complex
# valued coefficients whose real and imaginary parts specify the coordinates of a phasor in a complex plane
# angle of each phasor os quantized to one of the four quadrants, setting two bits of phase information
# repeated all across the iris with many wavelet sizes, freq, orientations to extract 2048 bits

# identifying in which quadrant of the complex plane each resultant phasor lies when a given area of iris is projected
# onto complex valued 2d gabor wavelets
# span an eight-fold range from 0.15 to 1.2mm on iris
# typical iris diam: 11mm
# typical pupil diam: 5mm

# hamming distance as measure of dissimilarity between any two irises
# two phase code bit vectors are denoted code a code b and whose mask bit vectors are denoted mask a mask b

# homogeneous rubber sheet model assigns to each point of the iris a pair of real coord where r is on unit interval [0, 1]
# and theta is angle [0, 2pi]

    #
    #
    # # https://www.geeksforgeeks.org/python/opencv-getgaborkernel-method/
    # # arguments for getGaborKernel
    # ksize = 31 #size of filter kernel (width, height) in pixels
    # sigma = 5 # standard deviation of Gaussian envelope
    # theta = 0  # orientation of filter in degrees
    # lambd = 10 #wavelength of the sinusoidal factor
    # gamma = 0.5 # spatial aspect ratio of filter
    # # psi = 0 # phase offset of filter in degrees, default is 0
    # # ktype = cv2.CV_32F # data type of filter coefficients, default is cv2.CV_32F
    #
    # # quadrature 2d gabor wavelet for real and imaginary
    # # psi = 0 for real because
    # # psi = pi/2 for imaginary because
    # k_real = cv2.getGaborKernel((ksize, ksize), sigma, 0, lambd, gamma, 0)
    # k_imaginary = cv2.getGaborKernel((ksize, ksize), sigma, 0, lambd, gamma, np.pi/2)
    #
    # # projection onto wavelet
    # re = cv2.filter2D(img, cv2.CV_8U, k_real)
    # ima = cv2.filter2D(img, cv2.CV_8U, k_imaginary)
    #