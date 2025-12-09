from weakref import ref
import numpy as np

def costFuncSAD(currentBlk, refBlk, mbSize):
    """Returns the Sum of Absolute Difference (SAD) between the two given blocks
    
        **Parameters:**

        - *currentBlk:* a 2D numpy array containing the pixel values of the current block
        - *refBlk:* a 2D numpy array containing the pixel values of the reference block
        - *mbSize:* size N of the NxN macroblocks 
        
        **Returns: **

        - *cost:* the computed SAD
    """
    
    err = 0
    cost = np.zeros((mbSize, mbSize))

    # --------- Modify Code Here ---------
    for i in range(mbSize):
        for j in range(mbSize):
            cost = abs(currentBlk[i][j] - refBlk[i][j])

    # ------------------------------------

    return err


# --------- Do Not Modify Below ---------
def main():
    """Main function to test costFuncSAD"""

    ref = np.array([[140, 124, 125, 131, 130, 138, 100,  87],
                    [139, 124, 126, 132, 134, 132,  88, 117],
                    [144, 125, 127, 135, 134, 138,  81,  82],
                    [149, 127, 127, 135, 137, 134,  79, 130],
                    [148, 128, 126, 138, 139, 144, 132, 144],
                    [147, 130, 122, 139, 138, 141, 144, 137],
                    [142, 136, 123, 137, 139, 139, 143, 111],
                    [139, 137, 125, 137, 140, 140, 149, 142]])

    cur = np.array([[140, 124, 124, 132, 130, 139, 102,  88],
                    [140, 123, 126, 132, 134, 134,  88, 117],
                    [143, 126, 126, 133, 134, 138,  81,  82],
                    [148, 126, 128, 136, 137, 134,  79, 130],
                    [147, 128, 126, 137, 138, 145, 132, 144],
                    [147, 131, 123, 138, 137, 140, 145, 137],
                    [142, 135, 122, 137, 140, 138, 143, 112],
                    [140, 138, 125, 137, 140, 140, 148, 143]])
    
    
    res = costFuncSAD(cur, ref, 8)
    
    # Expected result:
    print("cur =\t[")
    for i in cur:
        print("\t "+str(i))
    
    print("\t]")
    print("ref =\t[")
    for i in ref:
        print("\t "+str(i))
    
    print("\t]\ncostFuncSAD(cur, ref, 3) = "+ str(res))
    if round(res, 3) == 38:
        print('\033[92m' + "PASSED" + '\033[0m')
    else:
        print('\033[91m' + "FAILED => expected 38" + '\033[0m')

if __name__ == '__main__':
    
    main()