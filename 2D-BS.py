# Python code for binary search on sorted 2D array
def findBox(coords, target, num_rows, num_cols):
    # uses binary search to find the box with the correct target lat/long

    # row = len(arr)
    # col = len(arr[0])

    low, high = 0, num_rows*num_cols-1
 
    iter = 0

    while(low <= high):
        mid = low + (high-low)//2 # new index in array form
        mid_col = mid % num_cols
        mid_row = mid // num_cols

        # col = mid % col # this originally came before the row calculation
        # row = mid // col


        iter += 1
        print(f"Iteration #{iter}")

        print(f"row is: {mid_row}")
        print(f"col is: {mid_col}")
        print()

        box_top_right = coords[(mid_row, mid_col)][0]
        box_bottom_left = coords[(mid_row, mid_col)][1]

        if (box_bottom_left[0] <= target[0] <= box_top_right[0] and
            box_bottom_left[1] <= target[1] <= box_top_right[1]):
            return (mid_row, mid_col)

        # if within_box:
            # return (row, col)

        # left_of_box = target[1] < bottom_left[1]
        # if(left_of_box):
        #     low = mid + 1

        if target[0] < box_top_right[0] or target[1] < box_top_right[1]:
            low = mid + 1  # Target is to the right or above the box

        # if(val < target):
        #     low = mid + 1
        else:
            high = mid - 1
 
    return (-1, -1)

# Driver Code
if __name__ == '__main__':
    # Binary search in sorted matrix
    # arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    # box_info = {(i, j) : [(latitude, longitude) "for upper right", (latitude, longitude) "for bottom left"]}

    # box_info = {
    #     (0, 1): [(2, 2), (1, 1)],
    #     (0, 0): [(1, 2), (0, 1)],
    #     (1, 0): [(1, 1), (0, 0)],
    #     (1, 1): [(2, 1), (1, 0)],
    # }


    # ans = findBox(box_info, target=(1.2, 1.2), num_rows=2, num_cols=2)
    # print("Element found at indices: ", ans)

    box_info = {
        (0, 0): [(0, 2), (-1, 1)],
        (0, 1): [(1, 2), (0, 1)],
        (0, 2): [(2, 2), (1, 1)],
        (1, 0): [(0, 1), (-1, 0)],
        (1, 1): [(1, 1), (0, 0)],
        (1, 2): [(2, 1), (1, 0)],
        (2, 0): [(0, 0), (-1, -1)],
        (2, 1): [(1, 0), (0, -1)],
        (2, 2): [(2, 0), (1, -1)]
    }

    # Target within the specified box_info grid
    target = (-0.5, -0.2)  # This target should fall within box (2, 1)
    ans = findBox(box_info, target, num_rows=3, num_cols=3)
    print("Element found at indices: ", ans)