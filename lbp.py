import numpy as np
from skimage import feature
from skimage import color

def getLBPCoefficents(image, eps=1e-7):

    # Convert the input image to gray-scale
    rgb_image = color.lab2rgb(image)
    gray_image = color.rgb2gray(rgb_image)

    # Create image which will contains 
    # the LBPs variable of the input image
    imgLBP = np.zeros_like(gray_image)

    for ih in range(0, image.shape[0] - 3):

        for iw in range(0, image.shape[1] - 3):
            
            # Select a matrix 3x3 in image
            img = gray_image[ih:ih + 3, iw:iw + 3]
            center = img[1, 1]

            # Compute a new matrix
            # Equals 1 with pixels >= center
            # Equals 0 with pixels < center
            img01 = (img >= center)*1.0
            img01_vector = img01.T.flatten()

            img01_vector = np.delete(img01_vector, 4)
            
            # Convert binary line to decimal
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2**where_img01_vector)
            else:
                num = 0
            
            # Return center pixel as decimal
            imgLBP[ih+1, iw+1] = num

        # Compute histogram of the output image
        (hist, _) = np.histogram(imgLBP, bins=2**8)

        # Nomalize histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

    return(hist)
