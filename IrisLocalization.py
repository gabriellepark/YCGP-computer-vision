# IrisLocalization.py: detecting pupil and outer boundary of iris.

# Import functions
import cv2
import numpy as np
import matplotlib.pyplot as plt


# implementation based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1251145&utm_source=scopus&getft_integrator=scopus


def irisLocalization(img, show=False):
    """
    Function to locate iris to be passed into next steps
    This function uses the original image and detects pupil and outer boundary of iris.

    img = image we are normalizing.
    show (default = False): whether or not to show images. used for debugging and testing

    This methodology is based on Ma et al.

    """

    # Use of hough circles based on https://docs.opencv.org/4.13.0/da/d53/tutorial_py_houghcircles.html

    # Step 1: approximate pupil center
    # Blur image using a Gaussian blur because occlusion from eyelashes and eyelid was affecting the ability to detect the pupil
    # Based on opencv documentation: https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html#gae8bdcd9154ed5ca3cbc1766d960f45c1
    # ksize was selected to be wider than eyelashes but smaller than the pupil (most pupils in dataset are > 35px)
    blur = cv2.GaussianBlur(img, (31, 31), 0)

    # Project image in vertical and horizontal direction to approximately estimate the center coordinates of the pupil
    vert_proj = blur.sum(axis=0)
    horiz_proj = blur.sum(axis=1)

    # Get minima of two projection profiles
    Xp = int(np.argmin(vert_proj))
    Yp = int(np.argmin(horiz_proj))

    # Code to visualize the rough center
    if show:
        plt.imshow(img, cmap='gray')
        plt.plot(Xp, Yp, 'r+', markersize=15, mew=2)
        plt.title('Step 1: rough center')
        plt.show()

    # Step 2: refined center
    # Get 120 x 120 region of interest centered at point
    # This allows the ROI to be concentrated around an area where the pupil most likely is
    for i in range(2):
        x0, y0 = max(Xp - 120//2, 0), max(Yp - 120//2, 0)
        roi = img[y0:Yp + 120//2, x0:Xp + 120//2]

        # Binarize by adaptively selecting a reasonable threshold using the gray-level histogram of this region
        # Use Otsu thresholding to automatically calculate the optimal threshold by analyzing the image histogram
        # Based on: https://www.geeksforgeeks.org/python/otsu-thresholding-using-opencv/
        _, otsu_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Keep only the largest dark blob and drop regions with eyelashes
        # Documentation based on https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
        n, labels, stats, _ = cv2.connectedComponentsWithStats(otsu_thresh)
        # If number of labels is less than 2, return image
        if n < 2:
            return img, None, None
        # Find region that is the largest and apply a pupil mask over it
        biggest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        pupil_mask = (labels == biggest).astype(np.uint8)

        # Centroid of the binary region as accurate estimate of the pupil coordinates
        # Use image moments to calculate centroid https://docs.opencv.org/4.9.0/dd/d49/tutorial_py_contour_features.html
        m = cv2.moments(pupil_mask, binaryImage=True)
        Xp = int(x0 + m['m10'] / m['m00'])
        Yp = int(y0 + m['m01'] / m['m00'])

        # Rough pupil radius from the region's area
        r_pupil = np.sqrt(m['m00'] / np.pi)

    # Show plot of circle to visualize center and radius
    if show:
        plt.imshow(img, cmap='gray')
        plt.plot(Xp, Yp, 'r+', markersize=15, mew=2)
        plt.gca().add_patch(plt.Circle((Xp, Yp), r_pupil, color='lime', fill=False))
        plt.title('Step 2: refined center + radius')
        plt.show()

    # Step 3: Canny + Hough methods
    # Calculate the exact parameters of these two circles using edge detection
    # (Canny operator in experiments) and Hough transform in a certain region determined by the center of the pupil
    # Here, I use a wider region to capture both the pupil and iris
    # Earlier iterations with a smaller window were too small to capture the iris
    x0, y0 = max(Xp - 150, 0), max(Yp - 150, 0)

    # Based on: https://docs.opencv.org/4.13.0/da/d53/tutorial_py_houghcircles.html
    region = cv2.medianBlur(img[y0:Yp + 150, x0:Xp + 150], 5)

    # For pupil, search radius is around the step 2 estimate +/- 20% to ensure that anything that falls outside the radius is also captured
    pupil = cv2.HoughCircles(region, cv2.HOUGH_GRADIENT, 1, 300, param1=50, param2=15,
                             minRadius=int(int(r_pupil) * 0.8), maxRadius=int(int(r_pupil) * 1.2))

    # According to Daugman, the resolved iris radius is typically 80-130
    # for iris, we look for outer boundary radius range within 90-130 px range
    irises = cv2.HoughCircles(region, cv2.HOUGH_GRADIENT, 1, 5, param1=30, param2=20,
                              minRadius=90, maxRadius=130)
    if pupil is None or irises is None:
        return img, None, None

    # shift circle centers from region coordinates back to full image coordinates
    pupil = pupil[0, 0] + [x0, y0, 0]
    irises = irises[0] + [x0, y0, 0]

    # We want the strongest circle candidate that is concentric with the pupil
    # HoughCircles returns candidates sorted by votes, so the first close one is the strongest
    # If none are within 15 px, fall back to the circle whose center is closest to the pupil
    d = np.hypot(irises[:, 0] - pupil[0], irises[:, 1] - pupil[1])
    close = irises[d < 15]
    iris = close[0] if len(close) else irises[np.argmin(d)]

    # Show selected iris and pupil circls on original image
    if show:
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(cimg, (int(pupil[0]), int(pupil[1])), int(pupil[2]), (0, 255, 0), 1)
        cv2.circle(cimg, (int(iris[0]), int(iris[1])), int(iris[2]), (255, 255, 0), 1)
        cv2.imshow('Step 3: pupil (green), iris (cyan)', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img, pupil, iris
