# IrisLocalization.m/py: detecting pupil and outer boundary of iris. You can choose other iris
# localization methods if they work better
import cv2
import numpy as np
import matplotlib.pyplot as plt


# implementation based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1251145&utm_source=scopus&getft_integrator=scopus


# def irisLocalization(file_name, show=True):
def irisLocalization(img, show=True):

    # use of hough circles based on https://docs.opencv.org/4.13.0/da/d53/tutorial_py_houghcircles.html
    # img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # assert img is not None, "file could not be read, check with os.path.exists()"

    # Step 1 - approximate pupil center
    # project image in vertical and horizontal direction to approximately estimate the center
    # coordinates of the pupil
    vert_proj = img.sum(axis=0)
    horiz_proj = img.sum(axis=1)
    # get minima of two projection profiles
    Xp = int(np.argmin(vert_proj))
    Yp = int(np.argmin(horiz_proj))

    if show:
        plt.imshow(img, cmap='gray')
        plt.plot(Xp, Yp, 'r+', markersize=15, mew=2)
        plt.title('Step 1: rough center')
        plt.show()

    # Step 2 - refined center
    # get 120 x 120 region of interest centered at point
    x0, y0 = max(Xp - 60, 0), max(Yp - 60, 0)
    roi = img[y0:Yp + 60, x0:Xp + 60]

    # binarize by adaptively selecting a reasonable threshold using the gray-level histogram of this region
    # use Otsu thresholding to automatically calculate the optimal threshold by analyzing the image histogram
    # https://www.geeksforgeeks.org/python/otsu-thresholding-using-opencv/
    _, otsu_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # centroid of the binary region as accurate estimate of the pupil coordinates
    # use image moments to calculate centroid https://docs.opencv.org/4.9.0/dd/d49/tutorial_py_contour_features.html
    m = cv2.moments(otsu_thresh, binaryImage=True)
    Xp = int(x0 + m['m10'] / m['m00'])
    Yp = int(y0 + m['m01'] / m['m00'])

    # rough pupil radius from the region's area
    r_pupil = np.sqrt(m['m00'] / np.pi)

    if show:
        plt.imshow(img, cmap='gray')
        plt.plot(Xp, Yp, 'r+', markersize=15, mew=2)
        plt.gca().add_patch(plt.Circle((Xp, Yp), r_pupil, color='lime', fill=False))
        plt.title('Step 2: refined center + radius')
        plt.show()

    # Step 3: Canny + Hough
    # Calculate the exact parameters of these two circles using edge detection
    # (Canny operator in experiments) and Hough transform in a certain region determined by the center of the pupil.
    x0, y0 = max(Xp - 120, 0), max(Yp - 120, 0)
    # https://docs.opencv.org/4.13.0/da/d53/tutorial_py_houghcircles.html
    region = cv2.medianBlur(img[y0:Yp + 120, x0:Xp + 120], 5)

    # will come back to tune these params because it's not getting everything
    pupil = cv2.HoughCircles(region, cv2.HOUGH_GRADIENT, 1, 300, param1=50, param2=15,
                             minRadius=int(int(r_pupil) * 0.8), maxRadius=int(int(r_pupil) * 1.2))
    iris = cv2.HoughCircles(region, cv2.HOUGH_GRADIENT, 1, 300, param1=30, param2=20,
                            minRadius=int(r_pupil) + 20, maxRadius=120)

    if pupil is None or iris is None:
        return img, None, None

    pupil = pupil[0, 0] + [x0, y0, 0]
    iris = iris[0, 0] + [x0, y0, 0]

    if show:
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(cimg, (int(pupil[0]), int(pupil[1])), int(pupil[2]), (0, 255, 0), 1)
        cv2.circle(cimg, (int(iris[0]), int(iris[1])), int(iris[2]), (255, 255, 0), 1)
        cv2.imshow('Step 3: pupil (green), iris (cyan)', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img, pupil, iris

#
# if __name__ == '__main__':
#     irisLocalization('datasets/CASIA Iris Image Database (version 1.0)/001/1/001_1_1.bmp')

# if __name__ == '__main__':
#     base = 'datasets/CASIA Iris Image Database (version 1.0)'

#     for subject in range(1, 11):
#         s = f'{subject:03d}'
#         for session in (1, 2):
#             name = f'{s}_{session}_1.bmp'  # first image in the folder
#             print(name)
#             img, pupil, iris = irisLocalization(f'{base}/{s}/{session}/{name}', show=True)
