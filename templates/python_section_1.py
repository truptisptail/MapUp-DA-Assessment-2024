from typing import Dict, List
import pandas as pd

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    for i in range(0, len(lst), n):
        chunk = lst[i:i+n]
        result.extend(reversed(chunk))
    return result
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
       


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    dict = {}
    for string in lst:
        length = len(string)
        if length not in dict:
            dict[length] = []
        dict[length].append(string)
    return dict
print(group_by_length(["apple", "banana"]))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict[str, any]:
    flattened = {}
    def flatten_helper(sub_dict: Dict, parent_key: str = ''):
        for key, value in sub_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                flatten_helper(value, new_key)
            else:
                flattened[new_key] = value
                
    flatten_helper(nested_dict)
    return flattened
nested_dict_example = {
    'a': {
        'b': 1,
        'c': {
            'd': 2,
            'e': 3
        }
    },
    'f': 4
}
print(flatten_dict(nested_dict_example))
    

def unique_permutations(nums: List[int]) -> List[List[int]]:
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] not in seen:
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]
    result = []
    nums.sort()
    backtrack(0)
    return result
print(unique_permutations([1, 1, 2]))    


def is_valid_date(day: int, month: int, year: int) -> bool:
    if month < 1 or month > 12:
        return False
    if day < 1:
        return False
    days_in_month = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
                     31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return day <= days_in_month[month - 1]
def find_all_dates(text: str) -> List[str]:
    patterns = [
        r'(\d{2})-(\d{2})-(\d{4})',  
        r'(\d{2})/(\d{2})/(\d{4})',  
        r'(\d{4})\.(\d{2})\.(\d{2})'  
    ]
    valid_dates = []
    for pattern in patterns:
        matches = result.findall(pattern, text)
        for match in matches:
            if len(match) == 3:
                if pattern == patterns[0]:  
                    day, month, year = map(int, match)
                elif pattern == patterns[1]:  
                    month, day, year = map(int, match)
                else:  
                    year, month, day = map(int, match)
                if is_valid_date(day, month, year):
                    if pattern == patterns[0]:
                        valid_dates.append(f"{day:02d}-{month:02d}-{year}")
                    elif pattern == patterns[1]:
                        valid_dates.append(f"{month:02d}/{day:02d}/{year}")
                    else:
                        valid_dates.append(f"{year}.{month:02d}.{day:02d}")
    return valid_dates
text = "We met on 25-12-2021, and the deadline is 12/31/2022. My birthday is on 2023.01.01."
print(find_all_dates(text))

import math
def decode_polyline(polyline: str) -> list:
    coordinates = []
    index, lat, lng = 0, 0, 0
    while index < len(polyline):
        b, shift, result = 0, 0, 0
        while True:
            b = ord(polyline[index]) - 63
            result |= (b & 0x1f) << shift
            index += 1
            if b < 0x20:
                break
            shift += 5
        lat_change = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += lat_change    
        shift, result = 0, 0
        while True:
            b = ord(polyline[index]) - 63
            result |= (b & 0x1f) << shift
            index += 1
            if b < 0x20:
                break
            shift += 5
        lng_change = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += lng_change       
        coordinates.append((lat / 1e5, lng / 1e5))  
    return coordinates
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c 
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coordinates = decode_polyline(polyline_str)
    latitudes = []
    longitudes = []
    distances = []
    for i in range(len(coordinates)):
        latitudes.append(coordinates[i][0])
        longitudes.append(coordinates[i][1])
        if i > 0:
            dist = haversine(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
            distances.append(dist)
        else:
            distances.append(0) 
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance_m': distances
    })
    return df 
polyline_str = "u~ts~Fjz|r@?c@D"


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    if not matrix or not matrix[0]:
        return []
    rows, cols = len(matrix), len(matrix[0])
    rotated_matrix = [[0] * rows for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            rotated_matrix[c][rows - 1 - r] = matrix[r][c]
    transformed_matrix = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            index_sum = r + c 
            transformed_matrix[r][c] = rotated_matrix[r][c] * index_sum
    
    return []
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
result = rotate_and_multiply_matrix(matrix)
print(result)
    

def time_check(df) -> pd.Series:
    if 'timestamp' not in df.columns or 'id' not in df.columns or 'id_2' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp', 'id', and 'id_2' columns.")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    grouped = df.groupby(['id', 'id_2'])
    def check_group(group):
        min_time = group['timestamp'].min()
        max_time = group['timestamp'].max()
        if (max_time - min_time).days < 7:
            return False
        complete_range = pd.date_range(start=min_time, end=max_time, freq='H')
        return complete_range.isin(group['timestamp']).all()
    result = grouped.apply(check_group)
    
    return result
data = {
    'timestamp': [
        '2024-10-01 00:00', '2024-10-01 01:00', '2024-10-01 02:00',
        '2024-10-02 00:00', '2024-10-08 23:00', '2024-10-01 03:00',
        '2024-10-01 01:00', '2024-10-02 01:00', '2024-10-02 02:00'
    ],
    'id': [1, 1, 1, 1, 1, 2, 2, 2, 2],
    'id_2': [1, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
result = time_check(df)
print(result)