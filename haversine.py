import math

# def haversine_distance_david(predicted_lat, predicted_long, actual_lat, actual_long, in_meters=False): # predicted and true should be tuples
#     R = 6371 # Radius of the Earth in km

#     dLat = math.radians(predicted_lat - actual_lat)
#     dLon = math.radians(predicted_long - actual_long)
#     a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(actual_lat)) * math.cos(math.radians(predicted_lat)) * math.sin(dLon/2) * math.sin(dLon/2)
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
#     distance = R * c # Distance in km
#     return distance * 1000 if in_meters else distance # Convert to meters if required

# def haversine_distance_ashton(predicted_lat, predicted_long, actual_lat, actual_long, in_meters=False): # predicted and true should be tuples
#     R = 6371 # Radius of the Earth in km

#     predicted_lat = math.radians(predicted_lat)
#     predicted_long = math.radians(predicted_long)
#     actual_lat = math.radians(actual_lat)
#     actual_long = math.radians(actual_long)

#     delta_phi = actual_lat - predicted_lat
#     delta_lambda = actual_long - predicted_long
#     distance = 2 * R * math.asin(np.sqrt((math.sin(delta_phi/2))**2 + (math.cos(predicted_lat) * math.cos(actual_lat) * (np.sin(delta_lambda/2))**2)))
#     return distance * 1000 if in_meters else distance # Convert to meters if required

# def haversine_distance_claire(predicted_lat, predicted_long, actual_lat, actual_long, in_meters=False): # predicted and true should be tuples
#     # distance haversine 
#     R = 6371 # Radius of the Earth in km
#     rad = 2 * R 
#     # convert lat and longitudes to radians 
#     predicted_lat, predicted_long  = math.radians(predicted_lat), math.radians(predicted_long)
#     actual_lat, actual_long  = math.radians(actual_lat), math.radians(actual_long)
#     dlat = 1 - math.cos(actual_lat - predicted_lat)
#     dlon = math.cos(predicted_lat) * math.cos(actual_lat) * (1 - math.cos(actual_long - predicted_long))
#     distance = rad * math.asin(math.sqrt((dlat + dlon)/2))
#     return distance * 1000 if in_meters else distance # Convert to meters if required

def haversine_distance(predicted_lat, predicted_long, actual_lat, actual_long, in_meters=False): # predicted and true should be tuples
    # distance haversine 
    R = 6371 # Radius of the Earth in km
    rad = 2 * R 
    # convert lat and longitudes to radians 
    predicted_lat, predicted_long  = math.radians(predicted_lat), math.radians(predicted_long)
    actual_lat, actual_long  = math.radians(actual_lat), math.radians(actual_long)
    dlat = 1 - math.cos(actual_lat - predicted_lat)
    dlon = math.cos(predicted_lat) * math.cos(actual_lat) * (1 - math.cos(actual_long - predicted_long))
    distance = rad * math.asin(math.sqrt((dlat + dlon)/2))
    return distance * 1000 if in_meters else distance # Convert to meters if required