from gui import GUI


def weighted_estimate(grid, min_lat=24, max_lat=50, min_lon=-126, max_lon=-66, num_rows=15, num_cols=10):
    '''iterate through array likelihood each box'''
    total_lon = 0.0
    total_lat = 0.0
    
    lat_per_cell = (max_lat - min_lat) / num_rows
    lon_per_cell = (max_lon - min_lon) / num_cols

    find_mid_lat_long(row, col, lon_per_cell, lat_per_cell)
    total_area = num_rows*num_cols

    for box_index in range(total_area):
        row = box_index // num_cols
        col = box_index % num_cols
        
        weight = grid[box_index]
        # we're given the indices of the boxes, we now need to calculate the center of the box's latitutde and longitude            
        center_box_lat, center_box_long = find_mid_lat_long(row, col, lon_per_cell, lat_per_cell)
        total_lon += (weight * center_box_long)
        total_lat += (weight * center_box_lat)

    estimated_lon = total_lon/total_area
    estimated_lat = total_lat/total_area

    return estimated_lat, estimated_lon

def find_mid_lat_long(row, col, lon_per_cell, lat_per_cell, min_lon=-126, min_lat=24):
    center_box_long = (col * lon_per_cell) + min_lon + lon_per_cell/2
    center_box_lat = (row * lat_per_cell) + min_lat + lat_per_cell/2

    return center_box_lat, center_box_long

def main():
    min_lat=24
    max_lat=50
    min_lon=-126
    max_lon=-66

    lat_per_cell = (max_lat - min_lat) / 15
    lon_per_cell = (max_lon - min_lon) / 10

    i, j = find_mid_lat_long(7, 7, lon_per_cell, lat_per_cell)

    print(i, j)

    gui = GUI(num_rects_width=10, num_rects_height=15)
    gui.init()
    gui.clear_output()
    gui.place_dot(j, i, color='red', r=5)
    gui.show(display_coords=False, show_boxes=True)


# weighted_estimate()