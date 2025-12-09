import numpy as np
import cv2 as cv
import tools
from motionComp import motionComp
from minCost import minCost
from costFuncSAD import costFuncSAD
import time,sys

def motionEstimationFullSearch(currImg, refImg, mbSize, p):
    """Perform Full Search (FS) motion estimation algorithm on the given images
    
        **Parameters**:

        - *currImg:* a 2D array of the image to predict
        - *refImg:* a 2D array of the reference image
        - *mbSize:* size N of the NxN macroblocks
        - *p:* the parameter defining the search window
        
        ```
                                    . .. -p .. .
                                    . ..  . .. .
                                    . ..  . .. .
        search window for p=3 :    -p ..  x .. p
                                    . ..  . .. .
                                    . ..  . .. .
                                    . ..  p .. .
        ```                              
        
        **Returns:**

        - *mv:* a 2D numpy array containing the motion vectors of each macroblock between currImg and refImg
        - *complexity:* number of SAD computations performed while estimating the motion between currImg and refImg
    """
    
    if currImg.shape != refImg.shape:
        print("ERROR: images do not have the same dimensions")
        return -1
    
    [h,w] = currImg.shape
    
    float_lim = np.finfo(np.float16).max

    # Initialization of motion vectors array and costs mattrix
    mv = np.zeros((int(h*w/mbSize**2), 2), dtype=np.int8)
    costs = np.array([[float_lim for j in range(2*p + 1)] for i in range(2*p + 1)])
    
    idx_mb = 0
    complexity = 0
    
    # Starting from the top-left corner, perform motion estimation  within the search window for each macroblock
    for i in np.arange(0, h, mbSize):
        for j in np.arange(0, w, mbSize):
            
            # Full Search: 
            # For each possible displacement within the search window, estimate the cost of the prediction error
            # dy: vertical motion
            # dx: horizontal motion
            
            # ---------------------------------------------------------------- Modify Code Here ----------------------------------------------------------------
            for dy in np.arange(0,0,1):
                for dx in np.arange(0,0,1):
                    
                    # Coordinates of the top-left pixel of the reference block
                    i_refBlk = 0
                    j_refBlk = 0

                    costs[i+dx][j+dy] = currImg[i+dx][j+dy] - refImg[i+dx][j+dy]
                    
                    # Edge management - DO NOT CHANGE
                    if(i_refBlk < 0 or i_refBlk+mbSize > h or j_refBlk < 0 or j_refBlk+mbSize > w):
                        continue
                    
                    # Compute MAD for the tested reference block
                    costs[dy+p,dx+p] = 0
                    complexity += 1
                    
            # Now that the cost matrix is known, find the minimal cost and store the corresponding motion vector in the return array       
            [minRowIdx, minColIdx] = [0,0]
            mv[idx_mb][0] = 0
            mv[idx_mb][1] = 0
            # ------------------------------------------------------------------------------------------------------------------------------------------------

            # Preparation for next loop iteration
            idx_mb += 1
            costs = np.array([[float_lim for j in range(2*p + 1)] for i in range(2*p + 1)])
                    
    
    return [mv,complexity]


# --------- Do Not Modify Below ---------
def main():
    """Main function  used to test the motionEstimationFullSearch alogrithm"""

    mbSize = int(sys.argv[1])
    p = int(sys.argv[2])
    path = 'images/'
    images = ['paris_0001.bmp', 'paris_0002.bmp', 'paris_0003.bmp', 'paris_0004.bmp', 'paris_0005.bmp', 'paris_0006.bmp', 'paris_0007.bmp']

    overall_complexity = 0
    
    run_time = 0
    for idx in range(len(images)-1):

        loop_time = time.time()
        idx_refImg = idx
        idx_currImg = idx + 1

        # Read images form files
        refImg = cv.imread(path+images[idx_refImg], 0).astype(np.int16)
        currImg = cv.imread(path+images[idx_currImg], 0).astype(np.int16)

        # Perform Full Search motion estimation algorithm
        [mv, complexity] = motionEstimationFullSearch(currImg, refImg, mbSize, p)

        # Perform the motion compensation
        predImg = motionComp(refImg, mv, mbSize)
        diff = (((currImg.astype(np.int16) - predImg.astype(np.int16)))+128).astype(np.uint8)
        # Display images
        mvImg = tools.drawMVs(currImg, mv, mbSize)
        

        # Compute PSNR between predicted image and original image
        psnr = tools.computePSNR(currImg.astype(np.int16), predImg.astype(np.int16), 255)
        print("\n"+images[idx_currImg]+"= > PSNR = "+str(psnr)+ " dB")
        print(images[idx_currImg]+" => complexity = "+str(complexity))
        overall_complexity += complexity
        loop_time = round(time.time() - loop_time, 3)
        print("--- Image "+images[idx_currImg]+" estimated in "+str(loop_time)+" seconds (Full Search) ---")
        run_time += loop_time
        tools.show_images(images[idx_refImg], refImg.astype(np.uint8), images[idx_currImg], currImg.astype(np.uint8), 'Predicted Image', predImg, 'Motion Vectors', mvImg, 'Prediction Error', diff)

    print("\nFull Search => overall complexity = "+str(overall_complexity))
    print("Full Search => total motion estimation runtime = "+str(round(run_time, 3))+" sec")

if __name__ == '__main__':
    main()