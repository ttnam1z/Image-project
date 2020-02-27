import matplotlib.pyplot as plt
import skimage.color as colr
import numpy as np
from scipy.spatial.distance import cdist
import PIL.Image as Img

import warnings
warnings.simplefilter("error")


def image_display(inImg, label, chosen=0):
    # Separate image into K cluster image 
    # then display all images and grey image and chosen image.
    K = int(np.max(label))
    sep_imgs = np.zeros((label.shape[0], label.shape[1],3,K+1))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            # Separate input image into K cluster images
            sep_imgs[i,j,:,int(label[i,j])] = inImg[i,j,:]
    
    grey_img = label *255/4
    _ ,axs = plt.subplots(nrows= 1, ncols=(K+3), sharex=True, sharey=True,figsize=(8, 4))
    for i in  range(K+1):
        axs[i].imshow(sep_imgs[:,:,:,i])
    axs[K+1].imshow(grey_img,cmap='gray', vmin=0, vmax=255)
    axs[K+2].imshow(sep_imgs[:,:,:,chosen])
    plt.show()

def image_chosen(inImg, label, chosen=0):
    # Get chosen cluster image based on chosen label
    chs_img = np.zeros((label.shape[0], label.shape[1],3))
    
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j]==chosen:
                chs_img[i,j,:] = inImg[i,j,:]

    return chs_img

def image2matrix(inImg):
    # Transform image to matix for clustering
    # matrix:   row is index of point.
    #           column is features
    ma = {}
    ma['shape']=np.array([inImg.shape[0], inImg.shape[1]])
    ma['sc'] = np.zeros((inImg.shape[0]*inImg.shape[1], inImg.shape[2]))
    j = inImg.shape[1]
    for i in range(inImg.shape[0]):
        for k in range(inImg.shape[2]):
            ma['sc'][i*j:(i+1)*j,k] =inImg[i,:,k]
        
    return ma

def lbarray2image(label,shape):
    # Transform label array to label image
    img_lb = np.zeros(shape)
    j = shape[1]
    for i in range(shape[0]):
        img_lb[i,:] = label[i*j:(i+1)*j]
    return img_lb

def kmeans_init_centers(X, k):
    # Random K centers from X
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    dist = cdist(X, centers)
    # return index of the closest center
    return np.argmin(dist, axis = 1)

def kmeans_update_centers(X, labels, K, old_centers):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # take average
        if Xk.shape[0] != 0:
            centers[k,:] = np.mean(Xk, axis = 0)
        else:
            centers[k,:] = old_centers[k,:]
    return centers

def check_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))

def cus_kmeans(X, K):
    # Init K centers
    centers = [kmeans_init_centers(X, K)]
    label = []
    while True:
        # Calculate distance then get label of nearest cluster
        label = kmeans_assign_labels(X, centers[-1])
        # Update new centers 
        new_centers = kmeans_update_centers(X, label, K, centers[-1])
        # Check if centers set is unchanged
        if check_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
    return (centers, label)



def preprocess(filepath):

    # Read image from filepath
    # then resize and transform to 'lab' space
    image = {}
    img_r =Img.open(filepath)
    image_resize = img_r.resize((128, 128),Img.ANTIALIAS)
    image['rgb'] = np.array(image_resize)[:,:,0:3]
    image['lab'] = colr.rgb2lab(image['rgb'])

    # Using only ab
    ab = 1

    # K clustering (3)
    K = 3

    # Calculate clustering
    img_test = image2matrix(image['lab'][:,:,range(ab,3)])
    (centers, label) = cus_kmeans(img_test['sc'], K)

    # Use nearest neighbour to choose desease cluster
    feature = np.array([24.05894165, 28.99247693],ndmin=2)
    dist = cdist(feature, centers[-1])
    chosen = np.argmin(dist, axis = 1)
    img_lb = lbarray2image(label, img_test['shape'])
    #image_display(colr.lab2rgb(image['lab']),img_lb,chosen[0])
    
    # return image where label image equals chosen
    return image_chosen(image['lab'],img_lb,chosen[0])
