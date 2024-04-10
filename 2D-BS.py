def findBox(box_info, target, num_rows, num_cols):
    row = binarySearch_row(box_info, target, num_rows)
    print() # for visual breakage
    col = binarySearch_col(box_info, target, num_cols)

    print()

    print(f"row is: {row}")
    print(f"col is: {col}")

    print("we get here")

    if row != -1 and col != -1:
        print(f"Box found at row {row}, column {col}")
        return (row, col)
    else:
        print("Box not found")
        return -1


def binarySearch_row(box_info, target, num_rows):
    l, r = 0, num_rows-1
    
    while l <= r:
        mid = (l + r) // 2
        
        top_right, bottom_left = box_info[(mid, 0)] # col is 0

        if bottom_left[1] <= target[1] <= top_right[1]:
            print(f"got row: {mid}")
            return mid

        elif target[1] < bottom_left[1]:
            r = mid - 1
            print("elif is true")
 
        else:
            l = mid + 1
            print("else is true")
    return -1

def binarySearch_col(box_info, target, num_cols):
    l, r = 0, num_cols-1
    
    while l <= r:
        mid = (l + r) // 2
        
        top_right, bottom_left = box_info[(0, mid)] # row is 0

        if bottom_left[0] <= target[0] <= top_right[0]:
            print(f"got col: {mid}")
            return mid

        elif target[0] < bottom_left[0]:
            r = mid - 1
            print("elif is true")
 
        else:
            l = mid + 1
            print("else is true")
    return -1


# Driver Code
if __name__ == '__main__':
    # arr = [2, 3, 4, 10, 40]
    # x = 10

    box_info = {
        (0, 0): [(1, 1), (0, 0)],
        (0, 1): [(1, 2), (0, 1)],
        (1, 1): [(2, 2), (1, 1)],
        (1, 0): [(2, 1), (1, 0)],
    }
 
    # Function call
    result = findBox(box_info, (0.5, 1.5), 2, 2)
    if result != -1:
        print("Element is present at index", result)
