import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    """
    Load data from a CSV file.
    Parameters:
    - file_path: str, path to the CSV file.
    Returns:
    - DataFrame if file is found and successfully loaded, None otherwise.
    """
    try:
        df = pd.read_csv(file_path)
        print("Data successfully loaded.")
        return df
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None

def clean_and_prepare_data(df):
    """
    Prepares the data for analysis by cleaning and converting data types.
    """
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['FactValueNumeric', 'Period', 'Location'], inplace=True)
    df['FactValueNumeric'] = pd.to_numeric(df['FactValueNumeric'], errors='coerce')
    df['Period'] = pd.to_numeric(df['Period'], errors='coerce').astype('Int64')
    df.dropna(subset=['FactValueNumeric'], inplace=True)
    return df

def rank_countries_by_cases(df, year, column_name='FactValueNumeric', top_n=10):
    """
    Ranks countries based on the specified column for a given year.
    """
    df_year = df[df['Period'] == year]
    ranked = df_year.groupby('Location')[column_name].sum().sort_values(ascending=False)
    return ranked.head(top_n), ranked.tail(top_n)

# Load the data
data_path = '/Users/adelieplumasseau/Desktop/Python Final Project/data.csv'
data = load_data(data_path)

if data is not None:
    # Clean and prepare the data
    data = clean_and_prepare_data(data)

    # Perform statistical analysis
    column_to_analyze = 'FactValueNumeric'
    print(f"\nPerforming statistical analysis on: {column_to_analyze}")
    basic_stats = {
        'Minimum': data[column_to_analyze].min(),
        'Maximum': data[column_to_analyze].max(),
        'Average (Mean)': data[column_to_analyze].mean(),
        'Sum': data[column_to_analyze].sum(),
        'Count': data[column_to_analyze].count(),
        'Median': data[column_to_analyze].median(),
        'Mode': data[column_to_analyze].mode()[0],
        'Standard Deviation': data[column_to_analyze].std(),
        'Range': data[column_to_analyze].max() - data[column_to_analyze].min()
    }

    for stat, value in basic_stats.items():
        print(f"{stat}: {value}")

    # Visualization: Trend of NTD Cases Over Time
    plt.figure(figsize=(10, 6))
    yearly_data = data.groupby('Period')[column_to_analyze].sum()
    plt.plot(yearly_data.index, yearly_data.values, marker='o', linestyle='-', color='blue')
    plt.title('Trend of NTD Cases (2010-2021)')
    plt.xlabel('Year')
    plt.ylabel('Total NTD Cases')
    plt.grid(True)
    plt.show()

    # Rank countries for a specific year
    latest_year = data['Period'].max()
    top_countries, bottom_countries = rank_countries_by_cases(data, latest_year, column_to_analyze)

    # Visualization of Top and Bottom Countries
    plt.figure(figsize=(10, 6))
    top_countries.plot(kind='bar', color='skyblue')
    plt.title(f'Top Countries with Most NTD Cases in {latest_year}')
    plt.ylabel('NTD Cases')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 6))
    bottom_countries.plot(kind='bar', color='lightgreen')
    plt.title(f'Countries with Least NTD Cases in {latest_year}')
    plt.ylabel('NTD Cases')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Data loading failed.")
