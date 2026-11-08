import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# implementation bsaed on Ma et al. 2003
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1251145&utm_source=scopus&getft_integrator=scopus

# WILL COME BACK TO ADJUST AND COMMENT

def match_iris(X_train, y_train, X_test, metric = 'cosine', angles = (-9, -6, -3, 0, 3, 6, 9), use_lda=True, n_components = None):
    # after feature extraction, iris image represented as a feature vector of length 1,536
    y_train = np.array(y_train)
    angle_train = np.tile(angles, len(y_train) // len(angles))

    # fisher linear disc to reduce dimensionality of feature vector
    # feature vector f = W^TV
    # W is projection matrix
    # V is original feature vector
    if use_lda:
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto').fit(X_train, y_train)
        f_train = lda.transform(X_train)
        f_test = lda.transform(X_test)
    else:
        f_train = np.asarray(X_train)
        f_test = np.asarray(X_test)

    # rotation invariance: iris unwrapped at initial angles -9, -6, -3, 0, 3, 6, 9,
    # giving 7 templates per class (one center per class per angle)
    classes = np.unique(y_train)
    centers = []

    for c in classes:
        eye_centers =[]
        for a in np.unique(angle_train):
            mask = (y_train == c) & (angle_train == a)
            eye_centers.append(f_train[mask].mean(axis=0))
        centers.append(eye_centers)
    centers = np.array(centers)

    # nearest center classifier for classification in low dim feature space
    # predictions = []
    # for f in f_test:
    #     if metric == 'l1':
    #         d = np.abs(centers - f).sum(axis=-1)
    #     elif metric == 'l2':
    #         d = ((centers - f)**2).sum(axis=-1)
    #     else:
    #         d = 1 - (centers @ f) / (np.linalg.norm(centers, axis=-1) * np.linalg.norm(f))
    #
    #     d = d.min(axis = 1)
    #
    #     argmin = np.argmin(d)
    #
    #     predictions.append(classes[argmin])
    #
    # return np.array(predictions)

    predictions, dists = [], []
    for f in f_test:
        if metric == 'l1':
            d = np.abs(centers - f).sum(axis=-1)
        elif metric == 'l2':
            d = ((centers - f)**2).sum(axis=-1)
        else:
            d = 1 - (centers @ f) / (np.linalg.norm(centers, axis=-1) * np.linalg.norm(f))

        d = d.min(axis=1)              # best of the 7 rotations, per class
        dists.append(d)
        predictions.append(classes[np.argmin(d)])

    return np.array(predictions), np.array(dists), classes


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