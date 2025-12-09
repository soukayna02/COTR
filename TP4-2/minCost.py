import numpy as np


def minCost(costs):
    """Returns the row and column indices of the min value of the input cost 2D Array

        **Parameters:**

        - *costs:* a 2D numpy array containing cost values

        **Returns: **

        - *minRowIdx:* the row index of the min value
        - *minColIdx:* the column index of the min value
        - *minCostVal:* the min value
    """

    [height, width] = [len(costs), len(costs[0])]

    minCostVal = float("inf")
    minRowIdx = 0
    minColIdx = 0

    # --------- Modify Code Here ---------
    for i in range(len(costs)):
        for j in range(len(costs)):
            if costs[i][j] < a:
                a = costs[i][j]
                minRowIdx = i
                minColIdx = j
    # ------------------------------------

    return [minRowIdx, minColIdx]


# --------- Do Not Modify Below ---------
def main():
    """Main function to test minCost"""

    costs = np.array([[5, 7, 6], [9, 8, 1], [5, 0, 6], [9, 8, 1]])
    res = minCost(costs)

    # Expected result:
    print("costs =\t[")
    for i in costs:
        print("\t "+str(i))

    print("\t]\nminCost(costs) = " + str(res))
    if res == [2, 1]:
        print('\033[92m' + "PASSED" + '\033[0m')
    else:
        print('\033[91m' + "FAILED => expected [2,1,0]" + '\033[0m')


if __name__ == '__main__':

    main()
