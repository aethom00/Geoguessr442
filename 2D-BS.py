# Python code for binary search on sorted 2D array
def findBox(coords, target):
    row = len(arr)
    col = len(arr[0])
    l, h = 0, row*col-1
 
    while(l <= h):
        mid = l + (h-l)//2 # new index in array form
        col = mid % col
        row = mid // col

        top_right = coords[(row, col)][0]
        # top_left = coords[(row, col)][1]
        bottom_left = coords[(row, col)][2]
        # bottom_right = coords[(row, col)][3]

        within_top_right = target[0] <= top_right[0] and target[1] <= top_right[1]
        # within_top_left = target[0] <= top_left[0] and target[1] >= top_left[1]
        within_bottom_left = target[0] >= bottom_left[0] and target[1] >= bottom_left[1]
        # within_bottom_right = target[0] >= bottom_right[0] and target[1] <= bottom_right[1]

        within_box = within_top_right and within_bottom_left

        if within_box:
            return (row, col)

        # if(val == target):
        #     return [tR, tC]

        left_of_box = target[1] < bottom_left[1]

        if(left_of_box):
            l = mid + 1

        # if(val < target):
        #     l = mid + 1
        else:
            h = mid - 1
 
    return [-1, -1]

# Driver Code
if __name__ == '__main__':
    # Binary search in sorted matrix
    arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    box_info = {(i, j) : [(latitude, longitude), (latitude, longitude)]}

    ans = findBox(box_info, target=(latitude, longitude))
    print("Element found at indices: ", ans)