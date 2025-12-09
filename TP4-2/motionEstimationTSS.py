import numpy as np
import cv2 as cv
import tools
from motionEstimationFullSearch import motionEstimationFullSearch
from motionComp import motionComp
from minCost import minCost
from costFuncSAD import costFuncSAD
import time, sys

def motionEstimationTSS(currImg, refImg, mbSize):
    """Perform Three Step Search (TSS) motion estimation algorithm on the given images
    
        **Parameters**:

        - *currImg:* a 2D array of the image to predict
        - *refImg:* a 2D array of the reference image
        - *mbSize:* size N of the NxN macroblocks 

        **Returns:**

        - *mv:* a 2D numpy array containing the motion vectors of each macroblock between currImg and refImg
        - *complexity:* number of SAD computations performed while estimating the motion between currImg and refImg
    """
    
    if currImg.shape != refImg.shape:
        print("ERROR: images do not have the same dimensions")
        return -1
    
    [h,w] = currImg.shape
    # Number of steps, 3 for TSS
    nb_steps = 3
    float_lim = np.finfo(np.float16).max
    
    # Initialization of motion vectors array and costs mattrix
    mv = np.zeros((int(h*w/mbSize**2), 2), dtype=np.int8)
    costs = np.array([[float_lim for j in range(nb_steps)] for i in range(nb_steps)])
    
    idx_mb = 0
    complexity = 0

    
    # Starting from the top-left corner, perform motion estimation within the search window for each macroblock
    for i in np.arange(0, h, mbSize):
        for j in np.arange(0, w, mbSize):
            
            # Three Step Search: 
            # For each step, estimate the cost of the prediction error for the possible displacements, 
            # dy: vertical motion
            # dx: horizontal motion
            
            idx_iteration = nb_steps - 1

            # Initialize the coordinates of the center of the search window for the first step
            i_sw = i
            j_sw = j

            # TSS loop
            while(idx_iteration >= 0):
                
            # ---------------------------------------------------------------- Modify Code Here ----------------------------------------------------------------    

                # Definition of the search window for the current step
                stepSize = 2**idx_iteration

                # Initialization of the best motion vector for this step
                mv_step = [0,0]

                for dy in np.arange(-stepSize, stepSize+1, stepSize):
                    for dx in np.arange(-stepSize, stepSize+1, stepSize):
                        
                        # Coordinates of the top-left pixel of the current reference block
                        i_refBlk = 0
                        j_refBlk = 0
                        
                        # Edge management - DO NOT CHANGE
                        if(i_refBlk < 0 or i_refBlk+mbSize > h or j_refBlk < 0 or j_refBlk+mbSize > w):
                            continue
                        
                        # Compute MAD for the tested reference block
                        costs[int(dy/stepSize)+1,int(dx/stepSize)+1] = 0
                        complexity += 1
                    
                    
                # Now that the cost matrix of the current step is known, find the minimal cost for this step and its associated motion vector
                [minRowIdx, minColIdx] = [0,0]
                
                mv_step[0] = 0
                mv_step[1] = 0

                # Update the search window for the next step
                i_sw += 0
                j_sw += 0
                idx_iteration -= 1

                # Update the resulting motion vector for the current macroblock  
                mv[idx_mb][0] += 0
                mv[idx_mb][1] += 0
            # ------------------------------------------------------------------------------------------------------------------------------------------------    

            costs = np.array([[float_lim for c in l] for l in costs])
            idx_mb += 1
    
    return [mv,complexity]


# --------- Do Not Modify Below ---------
def main():
    """Main function  used to test the motionEstimationTSS alogrithm"""

    mbSize = int(sys.argv[1])

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

        # Perform TSS motion estimation algorithm
        [mv, complexity] = motionEstimationTSS(currImg, refImg, mbSize)
        [mv2, complexity2] = motionEstimationFullSearch(currImg, refImg, mbSize, 2)
        # Perform the motion compensation
        predImg = motionComp(refImg, mv, mbSize)
        diff = (((currImg.astype(np.int16) - predImg.astype(np.int16)))+128).astype(np.uint8)
        # Display images
        mvImg = tools.drawMVs(currImg, mv, mbSize)
        

        # Compute PSNR between predicted image and original image
        psnr = tools.computePSNR(currImg.astype(np.int16), predImg.astype(np.int16), 255)
        print(images[idx_currImg]+" => PSNR = "+str(psnr)+ " dB")
        print(images[idx_currImg]+" => complexity = "+str(complexity))
        overall_complexity += complexity
        loop_time = round(time.time() - loop_time, 3)
        print("--- Image "+images[idx_currImg]+" estimated in "+str(loop_time)+" seconds (TSS) ---")
        run_time += loop_time
        tools.show_images(images[idx_refImg], refImg.astype(np.uint8), images[idx_currImg], currImg.astype(np.uint8), 'Predicted Image', predImg, 'Motion Vectors', mvImg, 'Prediction Error', diff)

    print("TSS => overall complexity = "+str(overall_complexity))
    print("TSS => total motion estimation effective runtime = "+str(round(run_time, 3))+" sec")

if __name__ == '__main__':
    main()
    