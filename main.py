import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from skimage import color

from sklearn.metrics import accuracy_score
from lbp import getLBPCoefficents

import cv2, os
import ntpath
from imutils import paths
import joblib
from preprocess import preprocess

# INITILAZE DATA AND LABELS LIST FPR TRAINING

data_train = []
labels_train = []

CDir = os.getcwd()

# TRAINING PHASE
for imagePath in paths.list_images(os.path.join(CDir, 'dataset/training/')):
    
    # Get the image of poison part
    image = preprocess(imagePath)

    # Extract features for the image of poison part
    features = getLBPCoefficents(image)

    # Append features and labels of training-set to arrays
    head, _ = ntpath.split(imagePath)
    _, label  = ntpath.split(head)
    labels_train.append(label)
    data_train.append(features)

# TRAINING MODEL WITH SKLEARN LIBRARY
model = SVC(C=1e5, random_state=None)
model.fit(data_train, labels_train)

# ACCURANCY
print("Accuracy: %.2f %%" %(100*model.score(data_train, labels_train)))

# SAVE MODEL TO A FILE
filename = 'final_model.sav'
joblib.dump(model, filename)


# INITILAZE DATA AND LABELS LIST FPR TRAINING
data_test = []
labels_test = []

# TESTING PHASE
for imagePath in paths.list_images(os.path.join(CDir, 'dataset/testing/')):

    image = preprocess(imagePath)
    
    features = getLBPCoefficents(image)

    head, _ = ntpath.split(imagePath)
    _, label  = ntpath.split(head)
    labels_test.append(label)
    data_test.append(features)


# ACCURANCY OF TESTING PHASE
labels_pred = model.predict(data_test)

print("Accuracy: %.2f %%" %(100*accuracy_score(labels_test, labels_pred)))

# Calculate and plot confusion matrix
disp = plot_confusion_matrix(model, data_test, labels_test,
                                cmap=plt.cm.Blues,
                                normalize='true')

plt.show()
