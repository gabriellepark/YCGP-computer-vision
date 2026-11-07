#%%
from IrisLocalization import irisLocalization # Gabby functions
from IrisNormalization import normalizeIris # Yelene functions
from ImageEnhancement import enhanceImage # Yelene functions
from FeatureExtraction import extract_features # Gabby functions
# from IrisMatching import getFLD, runNCC # Fisher Linear discriminant and Nearest Center classifier functions
from PerformanceEvaluation import getCRR, getCRR_plot, getROC # CRR and ROC functions
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import numpy as np

# All codes in this project aim to replicate the work from the following paper:
# Ma et al., Personal Identification Based on Iris Texture Analysis, IEEE TRANSACTIONS ON
# PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 25, NO. 12, DECEMBER 2003

def load_data(folder, session=['1','2'], range = (1,109)):
    """
    Function to load the dataset and build our training dataframe. It imports the images for each subject,
    and each session, and adds them to a dataset with its source. We have an indicator column 'split' to differentiate
    train and test images. The dataframe will be used throughout to keep track of each step applied to the images.
    """

    # Create base dataframe where all data will be saved
    df = pd.DataFrame(columns=['source', 'split', 'original_image'])

    # Loop over each subject images per session and add them to the dataframe
    try:
        for sess in session:
            for subject in range:
                s = f'{subject:03d}'
                name = f"{s}/{session}"

                # Read in each image in the subject's session folder
                for f in os.listdir(folder+"/"+name):
                    # Ensures only bmp files are read
                    if f.endswith(".bmp"): 
                        filename = folder+"/"+name+"/"+f
                        print(filename)
                        img = cv2.imread(folder+"/"+name+"/"+f, cv2.IMREAD_GRAYSCALE)
                        assert img is not None, "file could not be read, check with os.path.exists()"
                        # Add image to our dataframe, with its source name 
                        df.loc[len(df.index)] = [f, None, img]

            # Save subject as a label ( will be used for prediction target)
            df['subject'] = s

            # Create an indicator to differentiate training (session 1) and test sets (session 2)
            df['split'] = np.where(df['split'].isna(), sess, df['split'])

    except Exception as e:
        print(f"Error occured on data import: {e}")

    return df

def main():

    # Load dataset
    base = 'datasets/CASIA Iris Image Database (version 1.0)'

    # Test on images on 1st folder first
    df = load_data(folder=base, session='1', range=(1,2))
    print(df.shape)

    # Apply localization algorithm on all the images in our training dataset
    results = [irisLocalization(image, show=True) for image in df['original_image']]

    # Unpack results and assign to corresponding columns
    _, df['pupil_info'], df['iris_info'] = zip(*results)

    print(df['pupil_info'])

    # Normalize images
    df['normalized_images'] = df.apply(lambda row: normalizeIris(row['original_image'], M=64, N=512, pupil_info=row['pupil_info'], iris_info=row['iris_info']), axis=1)

    # Enhance normalized images
    df['enhanced_images'] = df.apply(lambda row: enhanceImage(row['normalized_images']), axis=1)

    # Plot equalized image histogram
    # plotHist(img_histEqualization, bins=256, range = [0, 256])

    # Plot normalized images and enhanced ones side by side for comparison
    for norm, enh in zip(df['normalized_images'],df['enhanced_images']):
        plt.figure()
        #stacking images side-by-side
        res = np.hstack((norm,enh))
        plt.imshow(res, cmap='gray')

    # Extract features from enhanced images
    df['features'] = df.apply(lambda row: extract_features(row['enhanced_images']), axis = 1)

    print(df['features'][0])

    # Create train and test instances for CRR calculations
    df_train = df[df['split']=='1'].reset_index(drop=True)
    df_test = df[df['split']=='2'].reset_index(drop=True)

    # Set dimensionality of feature vectors to test
    V_dimensions = [-1, 20, 40, 80, 100, 120, 140, 160, 180, 200, 220]

    # Test different features dimensions
    results_crr = []


    for dim in V_dimensions:
        # Using -1 as default full feature set
        # If not -1, then we apply dimensionality reduction
        if dim != -1:
            pass

            # FLD = getFLD()
            # FLD.fit(df_train.drop(columns='subject), df_train['subject'])
            # df_train_projected = FLD.transform(df_train)

            # Apply nearest Center classifier
            # df_feature_centers = df_train_projected.groupby(['subject'])['features'].apply(lambda x: runNCC(x)).reset_index()

            # Transform test data to match train dimension
            # df_test_projected = FLD.transform(df_test)

        else:
            pass
            # Apply nearest Center classifier
            # df_feature_centers = df_train.groupby(['subject'])['features'].apply(lambda x: runNCC(x)).reset_index()

            # df_test_projected = df_test.copy()



        # Get the Vector matrices to compute distancees
        V_train_matrix = np.vstack(df_feature_centers['features'])
        V_test_matrix = np.vstack(df_test_projected['features'])

        print(V_train_matrix.shape)
        print(V_test_matrix.shape)

        # Calculate CRR for each metric

        crr_l1 = getCRR(df_feature_centers[['subject']], df_test_projected[['subject']], V_test_matrix, V_train_matrix, metric='L1')
        crr_l2 = getCRR(df_feature_centers[['subject']], df_test_projected[['subject']], V_test_matrix, V_train_matrix, metric='L2')
        crr_cos = getCRR(df_feature_centers[['subject']], df_test_projected[['subject']], V_test_matrix, V_train_matrix)

        results_crr.append((crr_l1, crr_l2, crr_cos))

    # Plot CRR curve
    crr_plot = getCRR_plot(results_crr, V_dimensions)
    crr_plot.show()

    # Best dimension is 200, with L1 = , L2 = , and Cosine = 
    crr_l1_og = results_crr[0][0]
    crr_l2_og = results_crr[0][1]
    crr_cos_og = results_crr[0][2]

    crr_l1_best = results_crr[9][0]
    crr_l2_best = results_crr[9][1]
    crr_cos_best = results_crr[9][2]

    # Build CRR table for best dimension and compare to original feature
    data_dict = {'Similarity measure':['L1', 'L2','Cosine'],
                     'CRR Original features':[crr_l1_og, crr_l2_og, crr_cos_og],
                     'CRR Reduced features':[crr_l1_best, crr_l2_best, crr_cos_best]}
    
    crr_table = pd.DataFrame(data_dict)
    print("CRR table:", crr_table)

    # get ROC curve with couple thresholds

    roc_table, roc_plot = getROC(df_feature_centers[['subject']], df_test_projected[['subject']], V_test_matrix, V_train_matrix, thresholds=(0.446, 0.472, 0.502))
    roc_plot.show()
    print("ROC table:", roc_table)

    return df

if __name__ == "__main__":
    main()
# %%
