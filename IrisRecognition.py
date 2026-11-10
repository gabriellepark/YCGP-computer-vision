#%%
from IrisLocalization import irisLocalization # Gaby functions
from IrisNormalization import normalizeIris # Yelene functions
from ImageEnhancement import enhanceImage # Yelene functions
from FeatureExtraction import extract_features # Gaby functions
from IrisMatching import get_templates, nearest_center # Fisher Linear discriminant and Nearest Center classifier functions
from PerformanceEvaluation import getCRR, getROC, getCRR_plot # CRR and ROC functions
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import numpy as np

# All codes in this project aim to replicate the work from the following paper:
# Ma et al., Personal Identification Based on Iris Texture Analysis, IEEE TRANSACTIONS ON
# PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 25, NO. 12, DECEMBER 2003

def load_data(folder, session=('1','2'), subjects = range(1,109)):
    """
    Function to load the dataset and build our training dataframe. It imports the images for each subject,
    and each session, and adds them to a dataset with its source. We have an indicator column 'split' to differentiate
    train and test images. The dataframe will be used throughout to keep track of each step applied to the images.
    folder = folder where all images are located
    session = indicator for folder containing session 1 of the eye capture or session 2. Session is used as 
              training data, while session 2 is used as test data.
    subjects = corresponds to each subject the eye belongs to, functions as our ID and target labels for
               predictions later in the code. Default tests full dataset of 108 eyes
    """

    # Return base dataframe where all data is saved
    # Loop over each subject images per session and add them to the dataframe
    rows = []
    try:
        for sess in session:
            for subject in subjects:
                s = f'{subject:03d}'
                name = f"{s}/{sess}"

                # Read in each image in the subject's session folder
                for f in os.listdir(folder+"/"+name):
                    # Ensures only bmp files are read
                    if f.endswith(".bmp"):
                        filename = folder+"/"+name+"/"+f
                        # print(filename)
                        img = cv2.imread(folder+"/"+name+"/"+f, cv2.IMREAD_GRAYSCALE)
                        assert img is not None, "file could not be read, check with os.path.exists()"
                        # Add image to our dataframe, with its source name
                        # Save subject as a label ( will be used for prediction target)
                        # Create an indicator to differentiate training (session 1) and test sets (session 2)

                        rows.append({'source': f, 'subject': subject, 'split': sess, 'original_image': img})
    except Exception as e:
        print(f"Error occured on data import: {e}")
    return pd.DataFrame(rows)

def main():

    # Load dataset
    base = 'datasets/CASIA Iris Image Database (version 1.0)'

    # Angles to be used to approximate rotation invariance and unwrapping the iris ring in extract_features() and get_templates()
    angles = (-9, -6, -3, 0, 3, 6, 9)


    # Load all images (session 1 = train, session 2 = test)
    # We used range=(1, 21) for a quick test
    df = load_data(folder=base)
    print(df.shape)

    # Apply localization algorithm on all the images in our dataset
    results = [irisLocalization(image, show=False) for image in df['original_image']]

    # Unpack results and assign to corresponding columns
    _, df['pupil_info'], df['iris_info'] = zip(*results)

    # Normalize images
    df['normalized_images'] = df.apply(lambda row: normalizeIris(row['original_image'], M=64, N=512, pupil_info=row['pupil_info'], iris_info=row['iris_info']), axis=1)

    # Enhance normalized images
    df['enhanced_images'] = df['normalized_images'].apply(enhanceImage)

    # Plot equalized image histogram
    # plotHist(img_histEqualization, bins=256, range = [0, 256])

    # Plot normalized images and enhanced ones side by side for comparison
    # for norm, enh in zip(df['normalized_images'],df['enhanced_images']):
    #     plt.figure()
    #     #stacking images side-by-side
    #     res = np.hstack((norm,enh))
    #     plt.imshow(res, cmap='gray')

    # Create train and test instances for CRR calculations
    df_train = df[df['split']=='1'].reset_index(drop=True)
    df_test = df[df['split']=='2'].reset_index(drop=True)

    # 7 rotated templates per training image, test images as-is

    # Train set
    # np.roll wraps columns around to give continuous ring
    # round(512 * a /360) converts a rotation angle a in degrees into the number of columns
    # axis = 1 rotates along the columns
    # Rotation for each enhanced image, for each angle
    X_train = np.array([extract_features(np.roll(enh, round(512 * a / 360), axis=1))
                        for enh in df_train['enhanced_images'] for a in angles])
    # Subject label repeated 7 times per training image for y_train
    y_train = np.repeat(df_train['subject'].to_numpy(), len(angles))
    # Test set
    # One feature vector per test image without rotation
    X_test = np.vstack(df_test['enhanced_images'].apply(extract_features).tolist())
    y_test = df_test['subject'].to_numpy()

    # Set dimensionality of feature vectors to test
    # -1 = original feature set
    V_dimensions = [-1, 10, 20, 40, 60, 80, 100, 107]

    # Test different features dimensions
    # Initialize array and set to hold results
    results_crr = []
    projected = {}

    # For each dimension in set of LDA dimensions
    for dim in V_dimensions:
        # Using -1 as default full feature set
        # If not -1, then we apply dimensionality reduction
        if dim != -1:
            # Fisher linear discriminant applied then one center per class per rotation angle
            f_test, template, t_labels = get_templates(X_train, y_train, X_test, angles, use_lda=True, n_components=dim)
        else:
            # Apply nearest center classifier on original features
            f_test, template, t_labels = get_templates(X_train, y_train, X_test, angles, use_lda=False)

        # For each subject, save the class centers and the test features as dfs
        df_feature_centers = pd.DataFrame({'subject': t_labels, 'features': list(template)})
        df_test_projected = pd.DataFrame({'subject': y_test, 'features': list(f_test)})

        # Get the Vector matrices to compute distances
        V_train_matrix = np.vstack(df_feature_centers['features'])
        V_test_matrix = np.vstack(df_test_projected['features'])

        # Nearest center classifier for each of the three metrics
        pred_l1 = nearest_center(V_test_matrix, V_train_matrix, df_feature_centers['subject'], metric='L1')
        pred_l2 = nearest_center(V_test_matrix, V_train_matrix, df_feature_centers['subject'], metric='L2')
        pred_cos = nearest_center(V_test_matrix, V_train_matrix, df_feature_centers['subject'])

        # Calculate CRR for each metric
        crr_l1 = getCRR(df_test_projected[['subject']], pred_l1)
        crr_l2 = getCRR(df_test_projected[['subject']], pred_l2)
        crr_cos = getCRR(df_test_projected[['subject']], pred_cos)

        # Append results to crr result list
        results_crr.append((crr_l1, crr_l2, crr_cos))
        # Save results to projected
        projected[dim] = (df_feature_centers, df_test_projected, V_train_matrix, V_test_matrix)

    # Plot CRR curve for reduced dimensions only
    crr_plot = getCRR_plot(pd.DataFrame(results_crr[1:], columns=['L1', 'L2', 'Cosine']), V_dimensions[1:])
    plt.legend(['L1', 'L2', 'Cosine'])
    plt.show()

    # Best dimension is highest cosine CRR among reduced dimensions
    best = int(np.argmax([r[2] for r in results_crr[1:]])) + 1
    print(f'best dimension: {V_dimensions[best]}')

    # Get CRR for original dimensions
    crr_l1_og = results_crr[0][0]
    crr_l2_og = results_crr[0][1]
    crr_cos_og = results_crr[0][2]

    # Get CRR for the best dimension with highest cosine CRR
    crr_l1_best = results_crr[best][0]
    crr_l2_best = results_crr[best][1]
    crr_cos_best = results_crr[best][2]

    # Build CRR table for best dimension and compare to original feature
    data_dict = {'Similarity measure':['L1', 'L2','Cosine'],
                     'CRR Original features':[crr_l1_og, crr_l2_og, crr_cos_og],
                     'CRR Reduced features':[crr_l1_best, crr_l2_best, crr_cos_best]}

    crr_table = pd.DataFrame(data_dict)
    print("CRR table: ", crr_table)

    # get ROC curve with couple thresholds (at the best dimension)
    df_feature_centers, df_test_projected, V_train_matrix, V_test_matrix = projected[V_dimensions[best]]
    roc_table, roc_plot = getROC(df_feature_centers[['subject']], df_test_projected[['subject']], V_test_matrix, V_train_matrix, thresholds=(0.446, 0.472, 0.502))
    print("ROC table: ", roc_table)

    plt.show()

    return df
if __name__ == "__main__":
    main()
# %%
