import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances, roc_curve

def getL1Sim(input_feature, template):
    """"
    Function to calculate the L1 similarity between two eyes.
    input_feature = features array of the test image
    template = features array of the train image
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist

    # Computes L1 similarity using sklearn Manhattan distance, referred to as 'cityblock' following scipy
    # reference names, and returns index of minimum distance.
    min_index = pairwise_distances_argmin(input_feature, template, metric='cityblock')

    # Get distances matrix for ROC
    similarity_dist = pairwise_distances(input_feature, template, metric='cityblock')
    return min_index, similarity_dist


def getL2Sim(input_feature, template):
    """"
    Function to calculate the L2 similarity between two eyes.
    input_feature = features array of the test image
    template = features array of the train image
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist

    # Computes L2 similarity, referred to as 'euclidean' following scipy reference names, and returns index of minimum distance.
    min_index = pairwise_distances_argmin(input_feature, template, metric='euclidean')

    # Get distances matrix for ROC
    similarity_dist = pairwise_distances(input_feature, template, metric='euclidean')

    return min_index, similarity_dist


def getCosineSim(input_feature, template):
    """"
    Function to calculate the Cosine similarity between two eyes.
    input_feature = features array of the test image
    template = features array of the train image
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist

    # Computes cosine similarity
    min_index = pairwise_distances_argmin(input_feature, template, metric='cosine')

    # Get distances matrix for ROC
    similarity_dist = pairwise_distances(input_feature, template, metric='cosine')

    return min_index, similarity_dist


def getCRR(test_y, predicted):
    """
    Given test labels and predicted labels from nearest center classifier, return the CRR for identification mode

    test_y = labels assigned to the test images ( dataframe format for easy alignment with prediction )
    predicted = predicted subject for each test image

    """

    # Assign prediction from the nearest center classifier
    test_y['predicted_subject'] = predicted

    # Calculate CRR
    correct_preds = sum(test_y['predicted_subject']==test_y['subject'])
    crr = correct_preds/test_y.shape[0]

    return crr


def getCRR_plot(CRR, V_dimension):
    """"
    Function to plot the CRR curve.
    CRR = Correct Recognition Rates list (y-axis)
    V_dimension = list of the different feature dimensions being tested
    """
    # Plot CRR curve
    plt.figure()
    plt.plot( V_dimension, np.round(np.array(CRR)*100), marker='o')
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct Recognition rate (%)')
    plt.title('CRR Curve')
    return plt.gcf()

def getROC(train_y, test_y, input_feature, template, thresholds=(0.446, 0.472, 0.502)):
    """
    Given train labels and test labels with corresponding indices, thresholds,
    and test feature array and template feature array, returns the ROC table based on chosen thresholds.
    Function also plots the ROC curve, using 100 thresholds in linear space between min and max cosine similarity
    distances.

    train_y = labels assigned to the train images ( dataframe format for easy alignment with prediction )
    test_y = labels assigned to the test images ( dataframe format for easy alignment with prediction )
    input_feature = features array of the test image 
    template = features array of the train image
    thresholds = the thresholds to use to assess FAR and FRR.

    FAR = False Acceptance Rate
    FRR = False Rejection Rate
    """
    # Create mask for FRR and FAR ~ reshape to match similarity distance dimensions
    train_labels = train_y['subject'].to_numpy().reshape(1, -1)
    test_labels = test_y['subject'].to_numpy().reshape(-1, 1)
    mask = (train_labels == test_labels)

    # Get similarity matrix
    _, similarity_distances = getCosineSim(input_feature,template)

    # Define true preds and wrong preds based on similarity distances
    true = similarity_distances[mask]
    wrong = similarity_distances[~mask]

    # Initialize arrays to store rates
    frr_list = []
    far_list = []

    # Loop over 100 thresholds within similarity distances range, to calculate
    # FAR and FRR at each and plot the ROC curve
    for t in np.linspace(similarity_distances.min(), similarity_distances.max(), 100):
        # False rejection rate are true preds above threshold
        frr = (true>=t).mean()
        frr_list.append(frr)

        # False acceptance rate are wrong preds below threshold
        far = (wrong<t).mean()
        far_list.append(far)

    # Plot ROC curve
    plt.figure()
    plt.plot(far_list, frr_list)
    plt.xlabel('False Acceptance Rate')
    # Make x scale log to match paper ( fig. 13)
    plt.xscale('log')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC Curve')

    # Build ROC table
    # Re-Initialize arrays to store rates
    frr_list = []
    far_list = []

    # Loop over the chosen thresholds for the ROC table and calculate FAR and FRR at each
    for t in thresholds:
        # False rejection rate are true preds above threshold
        frr = (true>=t).mean()
        frr_list.append(frr)

        # False acceptance rate are wrong preds below threshold
        far = (wrong<t).mean()
        far_list.append(far)

    data_dict = {'Threshold':thresholds,
                'False Acceptance Rate':far_list,
                'False Rejection Rate':frr_list}
    
    # Create ROC table with resulting FARs and FRRs
    roc_table = pd.DataFrame(data_dict)

    return roc_table, plt.gcf()
