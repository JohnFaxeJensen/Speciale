import requests
import pandas as pd


def get_surge_from_https(url: str) -> str:
    """Fetch surge data from a given HTTPS URL."""
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        raise


def proccess_surge_data(name: str, year: int) -> pd.DataFrame:
    #construct url
    url = f'https://surgedat.climate.lsu.edu/surge/getSurgeInfoByStormNameAndYear?storm_name={name}&year={year}'

    try:
        data = get_surge_from_https(url)
    except requests.RequestException as e:
        print(f"HTTP error occurred while fetching surge data for {name} {year}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    #convert to json then to dataframe

    column_names = ['Storm Name', 'Year', 'Lon', 'Lat','Storm_Tide_ft','Storm_Tide_m','Surge_ft','Surge_m','Datum','Location','Basin','State', 'remove1','remove2']

    df = pd.DataFrame(data, columns=column_names)
    df = df.drop(columns=['remove1','remove2'])

    #convert storm_tide_ft to numeric
    df['Storm_Tide_ft'] = pd.to_numeric(df['Storm_Tide_ft'], errors='coerce')
    df['Storm_Tide_m'] = pd.to_numeric(df['Storm_Tide_m'], errors='coerce')
    df['Surge_ft'] = pd.to_numeric(df['Surge_ft'], errors='coerce')
    df['Surge_m'] = pd.to_numeric(df['Surge_m'], errors='coerce')
    #Some entries has feet but no meter values, and some has meter but no feet values, so make sure both are filled
    df['Storm_Tide_m'] = df.apply(lambda row: row['Storm_Tide_ft'] * 0.3048 if pd.isna(row['Storm_Tide_m']) and not pd.isna(row['Storm_Tide_ft']) else row['Storm_Tide_m'], axis=1)
    df['Storm_Tide_ft'] = df.apply(lambda row: row['Storm_Tide_m'] / 0.3048 if pd.isna(row['Storm_Tide_ft']) and not pd.isna(row['Storm_Tide_m']) else row['Storm_Tide_ft'], axis=1)
    df['Surge_m'] = df.apply(lambda row: row['Surge_ft'] * 0.3048 if pd.isna(row['Surge_m']) and not pd.isna(row['Surge_ft']) else row['Surge_m'], axis=1)
    df['Surge_ft'] = df.apply(lambda row: row['Surge_m'] / 0.3048 if pd.isna(row['Surge_ft']) and not pd.isna(row['Surge_m']) else row['Surge_ft'], axis=1)
    return df

#get relevant hurricane names and years from another dataframe
hurricane_df = pd.read_excel(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\Aslak_data.xls", sheet_name = 'ATD of ICAT')
hurricane_df['lf_ISO_TIME'] = pd.to_datetime(hurricane_df['lf_ISO_TIME'], format="%Y-%m-%d %H:%M:%S")
hurricane_df['Year'] = hurricane_df['lf_ISO_TIME'].dt.year
#only get unique storm names and not storms that start with 'Storm'
hurricane_df['name'] = hurricane_df['name'].str.strip()
hurricane_df = hurricane_df[~hurricane_df['name'].str.startswith('Storm', na=False)]
#create tuple list of (name, year)
# hurricane_tuples = list(hurricane_df[['name', 'Year']].drop_duplicates().itertuples(index=False, name=None))
# print(hurricane_tuples)

# result_frame = pd.DataFrame()
# for name, year in hurricane_tuples:
#     print(f"Processing surge data for {name} {year}...")
#     surge_data_df = proccess_surge_data(name, year)
#     result_frame = pd.concat([result_frame, surge_data_df], ignore_index=True)
# result_frame.to_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\raw_tool_data\all_surge_data.csv", index=False)

#try to merge the new csv with the excel data
surge_data_df = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\raw_tool_data\all_surge_data.csv")
preproccessed_hurricane_df = pd.read_excel(r'C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\surge_data_inspection.xlsx')
for row, index in preproccessed_hurricane_df.iterrows():
    name = index['name'].strip()
    year = index['Year']
    matching_surge = surge_data_df[(surge_data_df['Storm Name'].str.strip() == name) & (surge_data_df['Year'] == year)]
    if not matching_surge.empty:
        #find the highest value within 2 degrees lat/lon in total
        lat = index['lf_lat']
        lon = index['lf_lon']
        matching_surge['lat_diff'] = (matching_surge['Lat'] - lat).abs()
        matching_surge['lon_diff'] = (matching_surge['Lon'] - lon).abs()
        matching_surge['total_diff'] = matching_surge['lat_diff'] + matching_surge['lon_diff']
        #filter to only those within 2 degrees total
        matching_surge = matching_surge[matching_surge['total_diff'] <= 2]
        #pick the highest one of surge_m and storm_tide_m

    if not matching_surge.empty:
        # Get indices of max values, handling NaN
        surge_idx = matching_surge['Surge_m'].idxmax()
        tide_idx = matching_surge['Storm_Tide_m'].idxmax()
        
        # Check if we got valid indices (not NaN)
        has_surge = pd.notna(surge_idx)
        has_tide = pd.notna(tide_idx)
        
        if has_surge and has_tide:
            best_surge = matching_surge.loc[surge_idx]
            best_tide = matching_surge.loc[tide_idx]
            best_data = best_tide if best_tide['Storm_Tide_m'] > best_surge['Surge_m'] else best_surge
        elif has_surge:
            best_data = matching_surge.loc[surge_idx]
        elif has_tide:
            best_data = matching_surge.loc[tide_idx]
        else:
            continue  # Skip if neither has valid data
        
        # Add values in new columns to preproccessed_hurricane_df
        prefix = 't_max'
        preproccessed_hurricane_df.at[row, f'{prefix}_Surge_m'] = best_data['Surge_m']
        preproccessed_hurricane_df.at[row, f'{prefix}_Surge_ft'] = best_data['Surge_ft']
        preproccessed_hurricane_df.at[row, f'{prefix}_Storm_Tide_m'] = best_data['Storm_Tide_m']
        preproccessed_hurricane_df.at[row, f'{prefix}_Storm_Tide_ft'] = best_data['Storm_Tide_ft']
        preproccessed_hurricane_df.at[row, f'{prefix}_Datum'] = best_data['Datum']
        preproccessed_hurricane_df.at[row, f'{prefix}_total_diff'] = best_data['total_diff']
preproccessed_hurricane_df.to_excel(r'C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\surge_data_inspection_updated.xlsx', index=False)