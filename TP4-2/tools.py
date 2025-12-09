import numpy as np
import cv2 as cv

def drawMVs(image, mv, mbSize):
    """Display motion vectors on an image
    
        **Parameters:**

        - *img:* a 2D numpy array containing the pixels of the image
        - *mv:* a numpy array containing the motion vectors to display
        - *mbSize:* size N of the NxN macroblocks
        
        **Returns:**

        - *mvImg:* a 2D numpy array of the image with the motion vectors displayed
    """
    
    [h,w] = image.shape
    img = np.array([[[j,j,j] for j in l] for l in image], dtype=np.uint8)
    
    
    lines = np.zeros((len(mv),2,1,2), dtype=np.int32)
    
    idx_mb = 0
    for i in np.arange(0, h, mbSize):
        for j in np.arange(0, w, mbSize):
            
            lines[idx_mb][0][0] = [j, i]
            lines[idx_mb][1][0] = [j+mv[idx_mb][1], i+mv[idx_mb][0]]
            idx_mb += 1
    
    return cv.polylines(img, lines, False, (0,0,255), 1)


def computePSNR(img1, img2, d=255):
    """Compute PSNR between 2 images
    
        **Parameters:**

        - *img1:* a 2D array of the image to predict
        - *img2:* a 2D array of the reference image
        - *d:* maximum pixel value in the images
                                       
        **Returns:**

        - *psnr:* the computed PSNR
    """
    if img1.shape != img2.shape:
        print("ERROR: images do not have the same dimensions")
        return -1
    
    [h,w] = img1.shape
    
    mse = np.sum(np.sum([[(img1[i,j]-img2[i,j])**2 for j in range(w)] for i in range(h)]))/(h*w)
    return round(10*np.log10(d**2/mse), 2)

def show_images(*argv):
    """Show multiple images in different windows"""
    for i in np.arange(0, len(argv), 2):
        cv.namedWindow(argv[i])        
        cv.moveWindow(argv[i], 100+375*int(i/4),50+600*(int(i/2)%2))  
        cv.imshow(argv[i], argv[i+1])
    while(True):
        k = cv.waitKey(33)
        if k == -1:  # if no key was pressed, -1 is returned
            continue
        else:
            break
    cv.destroyAllWindows()