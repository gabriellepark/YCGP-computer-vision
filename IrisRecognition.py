#%%
# from IrisLocalization import localizeIris # Gabby functions
from IrisNormalization import normalizeIris # Yelene functions
from ImageEnhancement import enhanceImage # Yelene functions
# from FeatureExtraction import load_data, localizeIris # Gabby functions
# from IrisMatching import getFLD, runNCC # Fisher Linear discriminant and Nearest Center classifier functions
# from PerformanceEvaluation import getCRR, getROC # CRR and ROC functions
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import numpy as np

# All codes in this project aim to replicate the work from the following paper:
# Ma et al., Personal Identification Based on Iris Texture Analysis, IEEE TRANSACTIONS ON
# PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 25, NO. 12, DECEMBER 2003

def main():

    # Load dataset

    img_path = './iris_test_data/images/'
    df = pd.DataFrame(columns=['source', 'original_image'])

    for i in range(1,6):
        filename = 'iris_0'+str(i)+'.png'
        img = os.path.join(img_path, filename)
        img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        df.loc[len(df.index)] = [filename, img_gray]


    localization_output = pd.read_csv('./iris_test_data/localization.csv')

    # Attach localization 

    df = df.merge(localization_output, how='left', left_on='source', right_on='image')
    print(df.columns)
    print(df.shape)

    df['pupil_info'] = list(zip(df['pupil_x'], df['pupil_y'], df['pupil_r']))
    df['iris_info'] = list(zip(df['iris_x'], df['iris_y'], df['iris_r']))


    # Get the coordinates for this iris

    # pupil_info, iris_info = df['images'].apply(localizeIris)


    # df['pupil_coords'] = pupil_info
    # df['iris_coords'] = iris_info


    # Normalize images
    df['normalized_images'] = df.apply(lambda row: normalizeIris(row['original_image'], M=64, N=512, pupil_info=row['pupil_info'], iris_info=row['iris_info']), axis=1)

    # Enhance normalized images
    df['enhanced_images'] = df.apply(lambda row: enhanceImage(row['normalized_images']), axis=1)

    # Plot equalized image histogram
    # plotHist(img_histEqualization, bins=256, range = [0, 256])

    for norm, enh in zip(df['normalized_images'],df['enhanced_images']):
        plt.figure()
        res = np.hstack((norm,enh)) #stacking images side-by-side
        plt.imshow(res, cmap='gray')

    # fisherLinearDiscriminant = getFLD(df) 
    # nearestCenterClassifier = runNCC(df) 
    # tableCRR = getCRR()
    # ROC_plot = getROC()

    
    # print("ROC curve:")
    # plt.show(ROC_plot)

    # print("CRR reseults:", tableCRR)

    return df

if __name__ == "__main__":
    main()
# %%
