# Import functions
import numpy as np

def normalizeIris(img, M, N, pupil_info, iris_info):
    """
    Function to normalize the iris region, based on localization output.
    This function uses the original image, and maps its pixel values to our desired standard output: N and M.
    The paper used output dimension 64 x 512, which would be N = 512 and M = 64.
    As such, the function returns a N x M numpy array.
    """

    # Create the output image ND array
    normImg = np.zeros((M,N))

    # Unpack localization coordinates (center_x, center_y, radius)


    # Pupil center (x,y) = (cxp, cyp); pupil radius = rps
    cxp = pupil_info[0]
    cyp = pupil_info[1]
    rp = pupil_info[2]

    # Iris outer boundary center (x,y) = (cxi, cyi); iris outer boundary radius = ri
    cxi = iris_info[0]
    cyi = iris_info[1]
    ri = iris_info[2]

    for (Y, X), value in np.ndenumerate(normImg):

        # Find the direction angle
        theta = 2 * np.pi * X/N
        
        # Get coordinates of the pupil boundary in the direction theta
        xp = cxp + rp * np.cos(theta)
        yp = cyp - rp * np.sin(theta)

        # Get coordinates of the outer boundary in the direction theta
        xi = cxi + ri * np.cos(theta)
        yi = cyi - ri * np.sin(theta)

        x = xp + ((xi - xp))*(Y/M)
        y = yp + ((yi - yp))*(Y/M)

        normImg[Y, X] = img[int(round(y)), int(round(x))]

    return normImg
