import numpy as np

def motionComp(refImg, mv, mbSize):
    """Returns the prediction image computed from the reference image and the given motion vectors
    
        **Parameters:**

        - *refImg:* a 2D numpy array containing the pixel values of the reference image
        - *mv:* a 2D numpy array containing the motion vectors of each macroblock of the image
        - *mbSize:* size N of the NxN macroblocks 
        
        **Returns:** 

        - *predImg:* the predicted image obtained after motion compensation
    """
    [h, w] = refImg.shape
    
    predImg = np.zeros((h, w), dtype=np.uint8)
    
    idxMB = 0
    
    for i in np.arange(0, h, mbSize):
        for j in np.arange(0, w, mbSize):
            
            # dy: vertical motion (from the current row)
            # dx: horizontal motion (from the current column) 
            [dy,dx] = mv[idxMB]
            
            # --------------------------- Modify Code Here ---------------------------

            # Coordinates of the top-left pixel of the reference block
            i_refBlk = 0
            j_refBlk = 0

            predImg[i+dx][j+dy] = refImg[i][j]

            # Compute predicted block
            predImg[i:i+mbSize, j:j+mbSize] = 0

            # ------------------------------------------------------------------------
            
            idxMB += 1

    return predImg

# --------- Do Not Modify Below ---------
def main():
    """Main function to test motionComp"""

    import cv2 as cv
    import tools
    
    img = cv.imread('images/paris_0001.bmp', 0) 
    [h,w] = img.shape
    
    [mv_h, mv_w] = [int(h/8), int(w/8)]
    
    mv = np.array([[0,0] for i in range(mv_h*mv_w)])
    
    # Fill Motion Vector array with [4,4] MVs, except on last row and column (null vectors)
    for i in range(len(mv)-mv_w):
        if i%mv_w != mv_w-1 :
            mv[i] = [4,4]
    
    
    predImg = motionComp(img, mv, 8)
    mvImg = tools.drawMVs(img, mv, 8)
    
    test = cv.imread('images/test_motionComp_mv_4_4.bmp', 0) 
    
    if np.array_equal(predImg, test):
        print('\033[92m' + "PASSED" + '\033[0m')
    else:
        print('\033[91m' + "FAILED" + '\033[0m')
    
    tools.show_images('paris_0001.bmp', img, 'Predicted Image', predImg, 'Error with expected prediction image', (test-predImg)**2+128, 'Motion Vectors', mvImg)

if __name__ == '__main__':
    
    main()