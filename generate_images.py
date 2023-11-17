import random
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

extreme_points = {'N': 50, 'S': 24, 'E': -66, 'W': -126}

def generate_random_coordinate():
    lat = random.uniform(extreme_points['S'], extreme_points['N'])
    long = random.uniform(extreme_points['W'], extreme_points['E'])
    return (long, lat)

    
def save_image(image_url, coordinate):
    # Ensure the 'Data' folder exists
    if not os.path.exists('Data'):
        os.makedirs('Data')

    # Format the filename using the coordinates
    filename = f"{coordinate[0]}, {coordinate[1]}.jpg"  

    # Full path to save the image
    file_path = os.path.join('Data', filename)

    # Download and save the image
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        # print(f"Saved image to {file_path}")
    else:
        print(f"Error downloading image: {response.status_code}, URL: {image_url}")


def find_images_in_bbox(bbox, token, limit=100):
    url = f'https://graph.mapillary.com/images?access_token={token}&fields=id&bbox={bbox}&limit={limit}&is_pano=false'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return [item['id'] for item in data['data']] if data.get('data') else []
    else:
        print(f"Error fetching data: {response.status_code}")
        return []
    
def get_image_url_from_id(image_id, token):
    url = f'https://graph.mapillary.com/{image_id}?access_token={token}&fields=id,thumb_1024_url'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('thumb_1024_url')
    else:
        print(f"Error fetching image URL: {response.status_code}")
        return None
    
def generate_image(gui, app_access_token):
    while True:
        i = random.randint(0, gui.num_rects_width - 1)
        j = random.randint(0, gui.num_rects_height - 1)
        top_left = gui.pixel_loc_to_lat_long(i * gui.square_amount[0], j * gui.square_amount[1])
        bottom_right = gui.pixel_loc_to_lat_long((i + 1) * gui.square_amount[0], (j + 1) * gui.square_amount[1])
        bbox = f"{top_left[1]},{bottom_right[0]},{bottom_right[1]},{top_left[0]}"

        image_ids = find_images_in_bbox(bbox, app_access_token)
        if image_ids:
            for image_id in image_ids:
                image_url = get_image_url_from_id(image_id, app_access_token)
                if image_url:
                    save_image(image_url, (top_left[1], top_left[0]))  # Coordinate format might need adjustment
                    # print(f"Image found and saved for bbox: {bbox}")
                    return