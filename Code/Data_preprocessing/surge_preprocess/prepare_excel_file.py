#the goal of this script is to use information from surgedat webside and global peak database to c
#create an excel file which can be manually checked
import pandas as pd
import os
from datetime import datetime
import re
import requests

def parse_storm_date(date_str, year):
    """
    Parse various storm date formats and return start and end dates.
    Handles formats like:
    - "Aug 27- Sep 15"
    - "8/17-8/26"
    - "17-Sep"
    - "Sep"
    - "nan"
    
    Returns: tuple of (start_date, end_date) as datetime objects or (None, None)
    """
    if pd.isna(date_str) or str(date_str).lower() == 'nan':
        return None, None
    
    date_str = str(date_str).strip()
    
    # Month name to number mapping
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    try:
        # Pattern 1: "Aug 27- Sep 15" or "Aug 27 - Sep 15" (two month names)
        if re.search(r'[A-Za-z]{3}.*?[A-Za-z]{3}', date_str) and '-' in date_str:
            parts = re.split(r'\s*-\s*', date_str)
            if len(parts) == 2:
                start_part = parts[0].strip()
                end_part = parts[1].strip()
                
                # Parse start date
                start_match = re.match(r'([A-Za-z]{3})\s+(\d{1,2})', start_part)
                if start_match:
                    start_month = month_map[start_match.group(1).lower()]
                    start_day = int(start_match.group(2))
                    start_date = datetime(year, start_month, start_day)
                else:
                    return None, None
                
                # Parse end date
                end_match = re.match(r'([A-Za-z]{3})\s+(\d{1,2})', end_part)
                if end_match:
                    end_month = month_map[end_match.group(1).lower()]
                    end_day = int(end_match.group(2))
                    end_date = datetime(year, end_month, end_day)
                else:
                    return None, None
                
                return start_date, end_date
        
        # Pattern 1b: "Aug 2-18" or "Sep 9 - 16" (single month name with two days)
        if re.match(r'^[A-Za-z]{3}\s+\d{1,2}\s*-\s*\d{1,2}$', date_str):
            match = re.match(r'([A-Za-z]{3})\s+(\d{1,2})\s*-\s*(\d{1,2})', date_str)
            if match:
                month = month_map[match.group(1).lower()]
                start_day = int(match.group(2))
                end_day = int(match.group(3))
                start_date = datetime(year, month, start_day)
                end_date = datetime(year, month, end_day)
                return start_date, end_date
        
        # Pattern 1c: "Sep21-22" (month name directly followed by days)
        if re.match(r'^[A-Za-z]{3}\d{1,2}-\d{1,2}$', date_str):
            match = re.match(r'([A-Za-z]{3})(\d{1,2})-(\d{1,2})', date_str)
            if match:
                month = month_map[match.group(1).lower()]
                start_day = int(match.group(2))
                end_day = int(match.group(3))
                start_date = datetime(year, month, start_day)
                end_date = datetime(year, month, end_day)
                return start_date, end_date
        
        # Pattern 1d: "30-31 Aug" (day-day month format)
        if re.match(r'^\d{1,2}-\d{1,2}\s+[A-Za-z]{3}$', date_str):
            match = re.match(r'(\d{1,2})-(\d{1,2})\s+([A-Za-z]{3})', date_str)
            if match:
                start_day = int(match.group(1))
                end_day = int(match.group(2))
                month = month_map[match.group(3).lower()]
                start_date = datetime(year, month, start_day)
                end_date = datetime(year, month, end_day)
                return start_date, end_date
        
        # Pattern 2: "8/17-8/26" or "8/17 - 8/26"
        if '/' in date_str and '-' in date_str:
            parts = re.split(r'\s*-\s*', date_str)
            if len(parts) == 2:
                start_match = re.match(r'(\d{1,2})/(\d{1,2})', parts[0].strip())
                end_match = re.match(r'(\d{1,2})/(\d{1,2})', parts[1].strip())
                
                if start_match and end_match:
                    start_date = datetime(year, int(start_match.group(1)), int(start_match.group(2)))
                    end_date = datetime(year, int(end_match.group(1)), int(end_match.group(2)))
                    return start_date, end_date
        
        # Pattern 2b: "9/25/03" (full date with 2-digit year)
        if '/' in date_str and date_str.count('/') == 2 and '-' not in date_str:
            match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', date_str)
            if match:
                month = int(match.group(1))
                day = int(match.group(2))
                year_part = int(match.group(3))
                # Handle 2-digit years (00-99)
                if year_part < 100:
                    year_part = 1900 + year_part if year_part < 50 else 2000 + year_part
                date = datetime(year_part, month, day)
                return date, date
        
        # Pattern 3: "17-Sep" or "17-Sept" (day-month)
        if re.search(r'^\d{1,2}-[A-Za-z]{3}', date_str):
            match = re.match(r'(\d{1,2})-([A-Za-z]{3})', date_str)
            if match:
                day = int(match.group(1))
                month = month_map[match.group(2).lower()]
                date = datetime(year, month, day)
                return date, date
        
        # Pattern 4: Month only (e.g., "Sep", "October")
        if re.match(r'^[A-Za-z]+$', date_str):
            month_abbr = date_str[:3].lower()
            if month_abbr in month_map:
                month = month_map[month_abbr]
                # Return first and last day of the month
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1)
                else:
                    end_date = datetime(year, month + 1, 1)
                end_date = pd.Timestamp(end_date) - pd.Timedelta(days=1)
                return start_date, end_date.to_pydatetime()
    
    except (ValueError, KeyError):
        pass
    
    return None, None

def generate_surge_data(df):
    try:
        surge_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\globalpeaksurgedb.csv", encoding='utf-8')
    except UnicodeDecodeError:
        surge_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\globalpeaksurgedb.csv", encoding='latin-1')

    # Remove any garbled characters from column names
    surge_data.columns = surge_data.columns.str.encode('ascii', 'ignore').str.decode('ascii')
    surge_data = surge_data[surge_data['Country'] == 'US']
    print('surge data shape after country filter: ', surge_data.shape)
    # Clean Storm Name values - remove non-ASCII characters
    surge_data['Storm Name'] = surge_data['Storm Name'].str.encode('ascii', 'ignore').str.decode('ascii').str.strip()
    surge_data = surge_data[surge_data['Year'] >= 1900]

    surge_data = surge_data[['Surge ID','Datum','Storm Name','Surge_m', 'Surge_ft','Storm_Tide_m', 'Storm_Tide_ft', 'Year', 'State', 'Lat', 'Lon', 'Storm Dates']]
    dates =surge_data['Storm Dates'].values.tolist()
    start_dates = []
    end_dates = []
    for i, date_str in enumerate(dates):
        year = surge_data.iloc[i]['Year']
        start_date, end_date = parse_storm_date(date_str, year)
        start_dates.append(start_date)
        end_dates.append(end_date)
    surge_data['Start_Date'] = start_dates
    surge_data['End_Date'] = end_dates
    surge_data['Storm_Dates'] = dates
    

    #merge on df and do left join, fill missing values manually afterwards
    df_for_surge_merge = df[['ATCF_ID','name', 'Year', 'lf_state', 'lf_lat', 'lf_lon', 'lf_ISO_TIME']]

    def append_none_values(lists_dict):
        """Helper function to append None to all tracking lists"""
        for list_obj in lists_dict.values():
            list_obj.append(None)
    
    def append_surge_values(surge_row, lists_dict):
        """Helper function to append surge data from a row"""
        lists_dict['datum'].append(surge_row['Datum'])
        lists_dict['surge_ms'].append(surge_row['Surge_m'])
        lists_dict['surge_ft'].append(surge_row['Surge_ft'])
        lists_dict['storm_tide_m'].append(surge_row['Storm_Tide_m'])
        lists_dict['storm_tide_ft'].append(surge_row['Storm_Tide_ft'])
        lists_dict['lat_lon_diffs'].append(surge_row['total_diff'])
        lists_dict['start_dates'].append(surge_row['Start_Date'])
        lists_dict['end_dates'].append(surge_row['End_Date'])
        lists_dict['storm_dates'].append(surge_row['Storm Dates'])
        lists_dict['Lat_db'].append(surge_row['Lat'])
        lists_dict['Lon_db'].append(surge_row['Lon'])

    #try different merge approach to find closest lat/lon
    result_lists = {
        'datum': [],
        'surge_ms': [],
        'surge_ft': [],
        'storm_tide_m': [],
        'storm_tide_ft': [],
        'lat_lon_diffs': [],
        'start_dates': [],
        'end_dates': [],
        'storm_dates': [],
        'Lat_db': [],
        'Lon_db': []
    }
    
    sketchy_matches = []  # Track matches with distance > 2.0
    print(len(df_for_surge_merge))
    for index,row in df_for_surge_merge.iterrows():
        storm_name = row['name']
        year = row['Year']
        state = row['lf_state']
        lat = row['lf_lat']
        lon = row['lf_lon']
        matching_surges = surge_data[(surge_data['Storm Name'] == storm_name) & (surge_data['Year'] == year)].copy()
        len_matching = len(matching_surges)
        matching_surges['lat_diff'] = (matching_surges['Lat'] - lat).abs()
        matching_surges['lon_diff'] = (matching_surges['Lon'] - lon).abs()
        matching_surges['total_diff'] = matching_surges['lat_diff'] + matching_surges['lon_diff']
        if len_matching == 0:
            #try matching unnamed storms by year and closest lat/lon
            try:
                matching_surges = surge_data[(surge_data['Year'] == year)].copy()
                if index == 1:
                    print(matching_surges)
                if len(matching_surges) == 0:
                    append_none_values(result_lists)
                    continue
                #test if lf_iso_time is within start and end date
                
                iso_time = row['lf_ISO_TIME']
                matching_surges['valid_date'] = matching_surges.apply(
                    lambda x: x['Start_Date'] <= iso_time <= x['End_Date'] if pd.notna(x['Start_Date']) and pd.notna(x['End_Date']) else False,
                    axis=1
                )
                matching_surges = matching_surges[matching_surges['valid_date'] == True]
                if len(matching_surges) == 0:
                    append_none_values(result_lists)
                    continue

                matching_surges['lat_diff'] = (matching_surges['Lat'] - lat).abs()
                matching_surges['lon_diff'] = (matching_surges['Lon'] - lon).abs()
                matching_surges['total_diff'] = matching_surges['lat_diff'] + matching_surges['lon_diff']
                min_idx = matching_surges['total_diff'].idxmin()
                if pd.isna(min_idx):
                    append_none_values(result_lists)
                    continue
                closest_surge = matching_surges.loc[min_idx]
                append_surge_values(closest_surge, result_lists)
                if(closest_surge['total_diff'] > 2.0):
                    sketchy_matches.append({
                        'storm_name': storm_name,
                        'year': year,
                        'hurricane_lat': lat,
                        'hurricane_lon': lon,
                        'surge_lat': closest_surge['Lat'],
                        'surge_lon': closest_surge['Lon'],
                        'total_diff': closest_surge['total_diff'],
                        'reason': 'Unnamed storm match - distance > 2.0'
                    })
                continue
            except Exception as e:
                append_none_values(result_lists)
                print(f"Error processing storm {storm_name} in year {year}: {e}")
                continue
        if len_matching == 1 and matching_surges.iloc[0]['total_diff'] <= 2.0:
            append_surge_values(matching_surges.iloc[0], result_lists)
            continue
        if len_matching == 1 and matching_surges.iloc[0]['total_diff'] > 2.0:
            sketchy_matches.append({
                'storm_name': storm_name,
                'year': year,
                'hurricane_lat': lat,
                'hurricane_lon': lon,
                'surge_lat': matching_surges.iloc[0]['Lat'],
                'surge_lon': matching_surges.iloc[0]['Lon'],
                'total_diff': matching_surges.iloc[0]['total_diff'],
                'reason': 'Named storm match - distance > 2.0'
            })
            append_surge_values(matching_surges.iloc[0], result_lists)
            continue
        if len_matching > 1:
            #calculate distance
            min_idx = matching_surges['total_diff'].idxmin()
            if pd.isna(min_idx):
                append_none_values(result_lists)
                continue
            closest_surge = matching_surges.loc[min_idx]
            if(closest_surge['total_diff'] > 2.0):
                sketchy_matches.append({
                    'storm_name': storm_name,
                    'year': year,
                    'hurricane_lat': lat,
                    'hurricane_lon': lon,
                    'surge_lat': closest_surge['Lat'],
                    'surge_lon': closest_surge['Lon'],
                    'total_diff': closest_surge['total_diff'],
                    'reason': 'Multiple matches (named storm) - closest distance > 2.0'
                })
            append_surge_values(closest_surge, result_lists)
        else:
            append_none_values(result_lists)
    df_for_surge_merge['Datum'] = result_lists['datum']
    df_for_surge_merge['Surge_m'] = result_lists['surge_ms']
    df_for_surge_merge['Surge_ft'] = result_lists['surge_ft']
    df_for_surge_merge['Storm_Tide_m'] = result_lists['storm_tide_m']
    df_for_surge_merge['Storm_Tide_ft'] = result_lists['storm_tide_ft']
    df_for_surge_merge['Lat_Lon_Diff'] = result_lists['lat_lon_diffs']
    df_for_surge_merge['Start_Date'] = result_lists['start_dates']
    df_for_surge_merge['End_Date'] = result_lists['end_dates']
    df_for_surge_merge['Storm_Dates'] = result_lists['storm_dates']
    df_for_surge_merge['Lat_db'] = result_lists['Lat_db']
    df_for_surge_merge['Lon_db'] = result_lists['Lon_db']
    #save as excel to inspect missing values

    #Put all relevant info into Storm_Tide_m
    for row_index, row in df_for_surge_merge.iterrows():
        if pd.isna(row['Storm_Tide_m']):
            #try Storm_Tide_ft
            if not pd.isna(row['Storm_Tide_ft']):
                df_for_surge_merge.at[row_index, 'Storm_Tide_m'] = float(row['Storm_Tide_ft']) * 0.3048


    
    # Export sketchy matches
    if sketchy_matches:
        sketchy_df = pd.DataFrame(sketchy_matches)
        sketchy_df.to_csv(r'C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\sketchy_matches.csv', index=False)
        print(f"\nFound {len(sketchy_matches)} sketchy matches (distance > 2.0)")
        print(sketchy_df)
    else:
        print("\nNo sketchy matches found!")
    surge_data_df = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\raw_tool_data\all_surge_data.csv")

    for row, index in df_for_surge_merge.iterrows():
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
            df_for_surge_merge.at[row, f'Surge_m'] = best_data['Surge_m']
            df_for_surge_merge.at[row, f'Surge_ft'] = best_data['Surge_ft']
            df_for_surge_merge.at[row, f'Storm_Tide_m'] = best_data['Storm_Tide_m']
            df_for_surge_merge.at[row, f'Storm_Tide_ft'] = best_data['Storm_Tide_ft']
            df_for_surge_merge.at[row, f'Datum'] = best_data['Datum']
            df_for_surge_merge.at[row, f'Lat_Lon_Diff'] = best_data['total_diff']
            df_for_surge_merge.at[row, f'Lat_db'] = best_data['Lat']
            df_for_surge_merge.at[row, f'Lon_db'] = best_data['Lon']
    df_for_surge_merge.to_excel(r'C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\surge_data_inspection_with_tool.xlsx', index=False)
    return df_for_surge_merge