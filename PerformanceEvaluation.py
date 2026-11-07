import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances, roc_curve

def getL1Sim(input_feature, template):
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
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
    
    # Computes L2 similarity, referred to as 'euclidean' following scipy reference names, and returns index of minimum distance.
    min_index = pairwise_distances_argmin(input_feature, template, metric='euclidean')

    # Get distances matrix for ROC
    similarity_dist = pairwise_distances(input_feature, template, metric='euclidean')

    return min_index, similarity_dist


def getCosineSim(input_feature, template):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
    
    # Computes cosine similarity
    min_index = pairwise_distances_argmin(input_feature, template, metric='cosine')

    # Get distances matrix for ROC
    similarity_dist = pairwise_distances(input_feature, template, metric='cosine')

    return min_index, similarity_dist


def getCRR(train_y, test_y, input_feature, template, metric='cosine'):
    """
    Given train labels and test labels with corresponding indices, decision metric,
    and test feature and template, returns the CRR based on chosen metric.
    Default metric is cosine similarity.
    """

    # Initialize minimum distance index
    min_index = None

    if metric=='L1':
        # Get index of minimum distance match based on L1 similarity
        min_index, _ = getL1Sim(input_feature, template)

    elif metric=='L2':
        # Get index of minimum distance match based on L2 similarity
        min_index, _ = getL2Sim(input_feature, template)

    else:
        # Get index of minimum distance match based on cosine similarity
        min_index, _ = getCosineSim(input_feature, template)

    # Assign prediction based on min distance
    test_y['predicted_subject'] = train_y['subject'].iloc[min_index].to_numpy()

    # Calculate CRR
    correct_preds = sum(test_y['predicted_subject']==test_y['subject'])
    crr = correct_preds/test_y.shape[0]

    return crr


def getCRR_plot(CRR, V_dimension):
    # Plot CRR curve
    plt.figure()
    plt.plot( V_dimension, round(CRR*100))
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct Recognition rate (%)')
    plt.title('CRR Curve')
    return plt.figure()

def getROC(train_y, test_y, input_feature, template, thresholds=(0.446, 0.472, 0.502)):
    """
    Function to plot the ROC curve given thresholds. 
    FAR = False Acceptance Rate
    FRR = False Rejection Rate
    """
    # Create mask for FRR and FAR ~ reshape to match similarity distance dimensions
    train_labels = train_y['subject'].to_numpy().reshape(1, -1)
    test_labels = test_y['subject'].to_numpy().reshape(-1, 1)
    mask = (train_labels == test_labels)

    # Get similarity matrix
    _, similarity_distances = getCosineSim(input_feature,template)

    # True preds 
    true = similarity_distances[mask]
    wrong = similarity_distances[~mask]

    # Initialize arrays to store rates
    frr_list = []
    far_list = []

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

    # Build ROC table for best dimension and compare to original feature

    # Re-Initialize arrays to store rates
    frr_list = []
    far_list = []

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
    
    roc_table = pd.DataFrame(data_dict)

    return roc_table, plt.figure()
