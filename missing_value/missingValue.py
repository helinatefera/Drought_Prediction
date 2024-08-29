import pandas as pd
import numpy as np

def calculate_sample_mean(df, station, element):
    # Filter the data for the specific station and element (e.g., PRECIP)
    station_data = df[(df['NAME'] == station) & (df['Element'] == element)]
    # Ensure all data is numeric
    station_data.iloc[:, 4:] = station_data.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
    # Calculate the mean for each month and then the mean across all months
    sample_mean = station_data.iloc[:, 4:].mean(axis=1).mean()
    return sample_mean

def get_neighbor_data(station, year, month, element='PRECIP'):
    
    for n in station:
        df = pd.read_excel("data.xlsx")
        # Selecting the data of station for the year 2021 for january
        df = df[(df['NAME'] == n) & (df['YEAR'] == year) & (df["Element"] == element)]
        # print(f"Data for {n} in {month} {year}:\n", df)
        print(df[month])


    # Check the available data after filtering
    # filtered_data = df[(df['NAME'] == station) & (df['YEAR'] == year) & (df['Element'] == element)]
    # print(f"Filtered data for {station} in {month} {year}:\n", filtered_data)
    
    # Get the observed value at the neighbor station for the specific month and year
    # neighbor_data = filtered_data[month]
    
    # if not neighbor_data.empty:
    #     return neighbor_data.values[0]
    # else:
    #     return None

# Load the data
df = pd.read_excel('data.xlsx')

# Target station and neighbors
target_station = 'Abomsa'
neighbors = ['Dangila', 'Bedele', 'Gatira']
year = 2023
month = 'Jan'

get_neighbor_data(neighbors, year, month)
# Calculate and print sample means
Ms = calculate_sample_mean(df, target_station, 'PRECIP')
print(f"Sample Mean for Target Station {target_station} (Ms): {Ms}")

for neighbor in neighbors:
    Mi = calculate_sample_mean(df, neighbor, 'PRECIP')
    Yi = get_neighbor_data(df, neighbor, year, month)
    print(f"Sample Mean for Neighbor {neighbor} (Mi): {Mi}")
    print(f"Data for Neighbor {neighbor} in {month} {year}: {Yi}")
