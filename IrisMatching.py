# IrisMatching.py: using Fisher linear discriminant for dimension reduction and nearest center classifier for classification;

# Import packages
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from PerformanceEvaluation import getL1Sim, getL2Sim, getCosineSim

# implementation bsaed on Ma et al. 2003
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1251145&utm_source=scopus&getft_integrator=scopus


def get_templates(X_train, y_train, X_test, angles=(-9, -6, -3, 0, 3, 6, 9),
                  use_lda=True, n_components=None):
    """
    Function to build the matching template based on Ma et al. Section 3.4.
    Projects features with Fisher linear discriminant and computes one class center per subject per rotation angle

    X_train: training feature vectors
    y_train: subject label for each row of X_train
    X_test: test feature vectors
    angles: initial unwrapping angles
    use_lda: whether to use LDA or not
    n_components: number of LDA dimensions
    return: f_test, template, template_labels

    """
    # after feature extraction, iris image represented as a feature vector of length 1,536
    # Change subject labels to array
    y_train = np.array(y_train)
    # Tag each training row with its rotation angle
    angle_train = np.tile(angles, len(y_train) // len(angles))

    # Fisher linear disc to reduce dimensionality of feature vector
    # Feature vector f = W^TV
    # W is projection matrix
    # V is original feature vector
    # If use_lda = true, then transform with LDA
    if use_lda:
        # Fisher linear discriminant
        # solver='eigen': solves the eigenvalue problem directly and allows shrinkage
        # shrinkage = 'auto': regularizes within-class scatter
        # n_components: number of dimensions kept
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=n_components).fit(X_train, y_train)
        # Project training and test features into the reduced space
        f_train = lda.transform(X_train)
        f_test = lda.transform(X_test)

        # Center both sets on the training mean so cosine similarity measures angles around the center of the data
        # Uses the training mean only
        mean = f_train.mean(axis=0)
        f_train, f_test = f_train - mean, f_test - mean
    else:
        # Else, use no dimensionality reductio
        f_train = np.asarray(X_train)
        f_test = np.asarray(X_test)


    # rotation invariance: iris unwrapped at initial angles -9, -6, -3, 0, 3, 6, 9,
    # giving 7 templates per class (one center per class per angle)
    classes = np.unique(y_train)
    centers = []

    # For each class at each angle, compute the class center used by nearest center classifier
    for c in classes:
        for a in angles:
            # Select the training rows for subject c at angle a
            mask = (y_train == c) & (angle_train == a)
            # Class center = mean feature vector of those rows
            centers.append(f_train[mask].mean(axis=0))

    # Stack centers into a matrix and include subject label for each template row
    template = np.array(centers)
    template_labels = np.repeat(classes, len(angles))
    # return  test features, template: class centers, and template_labels: subject
    return f_test, template, template_labels


def nearest_center(input_feature, template, label, metric='cosine'):
    """
    Nearest center classifier based on Ma et al. Section 3.4.
    Assign each test feature vector to the class of its closest template

    input_feature: features array of the test images
    template: class centers from get_templates
    label: subject label of each template row
    metric: similarity measure
    return: predicted subject label for each test image

    """

    # Initialize minimum distance index
    min_index = None

    if metric == 'L1':
        # Get index of minimum distance match based on L1 similarity
        min_index, _ = getL1Sim(input_feature, template)

    elif metric == 'L2':
        # Get index of minimum distance match based on L2 similarity
        min_index, _ = getL2Sim(input_feature, template)

    else:
        # Get index of minimum distance match based on cosine similarity
        min_index, _ = getCosineSim(input_feature, template)

    # Assign prediction based on min distance
    # closest template overall = min over each class's 7 rotated templates, then nearest class (Ma)
    return np.asarray(label)[min_index]


# ________________________________________________________________________
# Notes from Ma et al.
# Ma’ method: an iris image is represented as a feature vector of
# length 1,536. Dimension reduction: Fisher linear discriminant.
# The nearest center classifier is used for classification

# after feature extraction, iris image represented as a feature vector
# fisher linear disc to reduce dimensionality of feature vector
# nearest center classifier used for classification
# two popular methods: pca and fisher
# fisher increases class separability by considering both info of all samples and underlying structure of each class

# fisher searches for projected vectors that best disc diff classes in terms of maximizing ratio of between-class to within-class scatter

# feature vector f = W^TV
# W is projection matrix
# V is original feature vector

# nearest center classifier for classification in low dim feature space

# m = arg min dn(f, fi) where f and fi are feature vector of unknown sample and ith class
# dn(f, fi) denotes similarity measure
# want to obtain iris rep invariant to translation, scale, and rotation

# obtain approx rotation invariance by unwrapping iris ring at different initial angles
# -9, -6, -3, 0, 3, 6, 9
# 7 templates
# when matching input feature vector with templates of iris class, min of seven scores is taken as final matching dist