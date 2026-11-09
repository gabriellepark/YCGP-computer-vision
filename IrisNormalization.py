# Import functions
import numpy as np

def normalizeIris(img, M, N, pupil_info, iris_info):
    """
    Function to normalize the iris region, based on localization output.
    This function uses the original image, and maps its pixel values to our desired standard output: N and M.

    img = image we are normalizing.
    M = the desired output height of the normalized image
    N = the desired output width of the normalized image.
    pupil_info = the pupil information retrieved from the localization step: (center_x, center_y, radius)
    iris_info = the iris outer boundary information retrieved from the localization step: (center_x, center_y, radius)
    
    The paper used output dimension 64 x 512, which would be N = 512 and M = 64.
    As such, the function returns a M x N numpy array.
    """

    # # Create the output image ND array
    # normImg = np.zeros((M,N))

    # Unpack localization coordinates (center_x, center_y, radius)
    # Pupil center (x,y) = (cxp, cyp); pupil radius = rp
    cxp = pupil_info[0]
    cyp = pupil_info[1]
    rp = pupil_info[2]

    # Iris outer boundary center (x,y) = (cxi, cyi); iris outer boundary radius = ri
    cxi = iris_info[0]
    cyi = iris_info[1]
    ri = iris_info[2]

    # Loop over each position In(X,Y) in the output image and lookup the value to assign based
    # on the original image pixel values Io(x,y)
    # This loop mimics the process described in the paper, where projection system and equations are described.
    # (Ma et al., Personal Identification Based on Iris Texture Analysis, 3.2.2 Iris Normalization, (2))

    # Find the direction angle for each output column 
    theta = 2 * np.pi * np.arange(N) / N

        # Get coordinates of the pupil boundary in the direction theta
    xp = cxp + rp * np.cos(theta)
    yp = cyp - rp * np.sin(theta)

    # Get coordinates of the outer boundary in the direction theta
    xi = cxi + ri * np.cos(theta)
    yi = cyi - ri * np.sin(theta)

    # Calculate the position of the pixel value to replicate to the output
    x = xp + ((xi - xp))*(np.arange(M)/M)[:, None]
    y = yp + ((yi - yp))*(np.arange(M)/M)[:, None]
    
    # If position calculated is outside the boundary of the image ( <0 or greater than max width or max height)
    # Ensures position remains at the edges, within the image
    x = np.clip(np.round(x), 0, img.shape[1] - 1).astype(int)
    y = np.clip(np.round(y), 0, img.shape[0] - 1).astype(int)

    # Look-up pixel value in original image to assign position in output image.
    # This is our In(X,Y) = Io(x,y) from the paper
    return img[y, x].astype(float)