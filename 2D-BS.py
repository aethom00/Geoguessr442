from gui import GUI

def find_grid_index(target_lat, target_lon, min_lat, max_lat, min_lon, max_lon, num_rows, num_cols):
    # Calculate the span of each cell
    lat_per_cell = (max_lat - min_lat) / num_rows
    lon_per_cell = (max_lon - min_lon) / num_cols
    
    # Calculate the indices
    i = int((target_lat - min_lat) / lat_per_cell)
    j = int((target_lon - min_lon) / lon_per_cell)
    
    # Adjust if on the edge
    i = min(i, num_rows - 1)
    j = min(j, num_cols - 1)
    
    return i, j

# our map settings
min_lat, max_lat = 24, 50
min_lon, max_lon = -126, -66
num_rows, num_cols = 15, 10

target_lat, target_long = 44.76306000, -85.62063000

i, j = find_grid_index(target_lat, target_long, min_lat, max_lat, min_lon, max_lon, num_rows, num_cols)
print(f"The target is in the box at indices: ({i}, {j})")

gui = GUI(num_rects_width=10, num_rects_height=15)
gui.init()
gui.clear_output()
gui.place_dot(target_long, target_lat, color='red', r=5)
gui.show(display_coords=False, show_boxes=True)
