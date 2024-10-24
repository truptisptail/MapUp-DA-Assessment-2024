import pandas as pd
import numpy as np 
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  
def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError("DataFrame must contain 'latitude' and 'longitude' columns.")
    num_points = len(df)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                lat1, lon1 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
                lat2, lon2 = df.iloc[j]['latitude'], df.iloc[j]['longitude']
                distance_matrix[i][j] = haversine(lat1, lon1, lat2, lon2)

    return pd.DataFrame(distance_matrix, index=df.index, columns=df.index)
data = {
    'latitude': [34.0522, 36.1699, 40.7128],
    'longitude': [-118.2437, -115.1398, -74.0060]
}
df = pd.DataFrame(data)
distance_matrix = calculate_distance_matrix(df)
print(distance_matrix)


def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    num_points = distance_matrix.shape[0]
    id_start = []
    id_end = []
    distances = []
    for i in range(num_points):
        for j in range(num_points):
            if i != j: 
                id_start.append(i)  
                id_end.append(j)
                distances.append(distance_matrix.iloc[i, j])
    unrolled_df = pd.DataFrame({
        'id_start': id_start,
        'id_end': id_end,
        'distance': distances
    })
    return unrolled_df
distance_data = {
    0: [0, 100, 200],
    1: [100, 0, 150],
    2: [200, 150, 0]
}
distance_matrix = pd.DataFrame(distance_data)
unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)

   

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    reference_distances = df[df['id_start'] == reference_id]['distance']
    reference_average = reference_distances.mean()
    threshold_low = reference_average * 0.9
    threshold_high = reference_average * 1.1
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
    within_threshold = avg_distances[
        (avg_distances['distance'] >= threshold_low) &
        (avg_distances['distance'] <= threshold_high)
    ]
    result = within_threshold.merge(df[['id_start']].drop_duplicates(), on='id_start')

    return result
data = {
    'id_start': [1, 1, 2, 2, 3, 3],
    'id_end': [2, 3, 1, 3, 1, 2],
    'distance': [100, 150, 110, 140, 90, 130]
}
df = pd.DataFrame(data)
result_df = find_ids_within_ten_percentage_threshold(df, reference_id=1)
print(result_df)


def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    toll_rates = {
        'car': 0.05,       
        'truck': 0.10,     
        'bus': 0.08,       
        'motorcycle': 0.03 
    }
    if 'vehicle_type' not in df.columns or 'distance' not in df.columns:
        raise ValueError("DataFrame must contain 'vehicle_type' and 'distance' columns.")
    df['toll_rate'] = df.apply(
        lambda row: row['distance'] * toll_rates.get(row['vehicle_type'], 0),
        axis=1
    )
    return df
data = {
    'vehicle_type': ['car', 'truck', 'bus', 'motorcycle', 'car'],
    'distance': [100, 200, 150, 80, 250]
}
df = pd.DataFrame(data)
result_df = calculate_toll_rate(df)
print(result_df)


def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    toll_rates = {
        'peak': 0.10,        
        'off_peak': 0.05,    
        'night': 0.03       
    }
    if 'timestamp' not in df.columns or 'distance' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' and 'distance' columns.")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    def get_toll_rate(row):
        hour = row['timestamp'].hour
        if 7 <= hour < 9 or 17 <= hour < 19:  
            return toll_rates['peak']
        elif 9 <= hour < 17:  
            return toll_rates['off_peak']
        else:  
            return toll_rates['night']
    df['toll_rate'] = df.apply(lambda row: get_toll_rate(row) * row['distance'], axis=1)
    
    return df
data = {
    'timestamp': [
        '2024-10-01 08:00', 
        '2024-10-01 10:00', 
        '2024-10-01 18:00', 
        '2024-10-01 23:00'
    ],
    'distance': [100, 200, 150, 80]
}
df = pd.DataFrame(data)
result_df = calculate_time_based_toll_rates(df)
print(result_df)