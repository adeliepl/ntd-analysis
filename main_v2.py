import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import scipy.stats as stats

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file with error handling.
    
    Parameters:
    - file_path: str, path to the CSV file.
    
    Returns:
    - DataFrame if file is successfully loaded.
    - None if file is not found or unreadable.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info("✅ Data successfully loaded. Shape: %s", df.shape)
        return df
    except FileNotFoundError:
        logging.error("❌ File not found at path: %s", file_path)
        return None
    except pd.errors.ParserError:
        logging.error("❌ Error parsing CSV. Check file format.")
        return None
    except Exception as e:
        logging.error("❌ Unexpected error: %s", str(e))
        return None

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by:
    - Removing empty columns
    - Converting numeric fields properly
    - Standardizing country names
    
    Parameters:
    - df: DataFrame, raw dataset
    
    Returns:
    - Cleaned DataFrame
    """
    # Drop empty columns
    df = df.dropna(axis=1, how="all")

    # Convert 'Value' column (formatted numbers) to numeric
    if "Value" in df.columns:
        df["Value"] = df["Value"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Ensure 'FactValueNumeric' is numeric
    if "FactValueNumeric" in df.columns:
        df["FactValueNumeric"] = pd.to_numeric(df["FactValueNumeric"], errors="coerce")

    return df


def perform_statistical_analysis(df, column="FactValueNumeric"):
    """
    Performs statistical analysis on the dataset.

    Parameters:
    - df: pd.DataFrame, cleaned dataset
    - column: str, the column containing NTD case counts

    Returns:
    - Prints statistical results
    """

    if column not in df.columns:
        logging.error("❌ Column '%s' not found in dataset.", column)
        return

    # Basic Descriptive Statistics
    mean_value = df[column].mean()
    median_value = df[column].median()
    mode_value = df[column].mode()[0] if not df[column].mode().empty else "No mode"
    std_dev = df[column].std()
    skewness = stats.skew(df[column], nan_policy='omit')
    kurtosis = stats.kurtosis(df[column], nan_policy='omit')
    min_value = df[column].min()
    max_value = df[column].max()

    # Correlation with Period (Year)
    if "Period" in df.columns:
        df.loc[:, "Period"] = pd.to_numeric(df["Period"], errors="coerce")
        correlation = df[[column, "Period"]].corr().iloc[0, 1]
    else:
        correlation = "N/A"

    # Print results properly formatted
    print("\n📊 **Statistical Analysis Report** 📊")
    print(f"✅ Mean: {mean_value:,.2f}")
    print(f"✅ Median: {median_value:,.2f}")
    print(f"✅ Mode: {mode_value}")
    print(f"✅ Standard Deviation: {std_dev:,.2f}")
    print(f"✅ Skewness: {skewness:.2f} (Distribution shape)")
    print(f"✅ Kurtosis: {kurtosis:.2f} (Tail heaviness)")
    print(f"✅ Minimum: {min_value:,.2f}")
    print(f"✅ Maximum: {max_value:,.2f}")
    print(f"✅ Correlation with Time (Period): {correlation:.2f}\n")

    return {
        "Mean": mean_value,
        "Median": median_value,
        "Mode": mode_value,
        "Standard Deviation": std_dev,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Minimum": min_value,
        "Maximum": max_value,
        "Correlation with Time": correlation
    }

def visualize_top_countries(df: pd.DataFrame, column: str = "FactValueNumeric", top_n: int = 10):
    """
    Creates a clean, professional bar chart for the top N countries requiring treatment.
    
    Parameters:
    - df: pd.DataFrame, cleaned dataset
    - column: str, the column to use for ranking (default is "FactValueNumeric")
    - top_n: int, number of top countries to display
    
    Returns:
    - Displays a well-styled bar chart.
    """
    # Aggregate data and get the top countries
    ranked_df = df.groupby("Location")[column].sum().sort_values(ascending=False).head(top_n)
    
    # Create figure with better width to avoid cut-off text
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")

    # Bar plot with warm color palette
    ax = sns.barplot(
        x=ranked_df.values, 
        y=ranked_df.index, 
        palette="magma_r",  # A warm, professional gradient
        hue=ranked_df.index, 
        edgecolor="black"
    )

    # Titles & labels with adjusted font sizes
    plt.title(f"Top {top_n} Countries Requiring Treatment", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Total Treatment Needs", fontsize=14, labelpad=15)
    plt.ylabel("Country", fontsize=14, labelpad=15)

    # Improve grid styling for subtle elegance
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.grid(False)

    # Remove unnecessary borders
    sns.despine(left=True, bottom=True)

    # Adjust text annotations outside the bars to prevent overlap
    for index, value in enumerate(ranked_df.values):
        ax.text(value + (value * 0.01), index, f"{int(value):,}", 
                va="center", fontsize=12, fontweight="bold", color="black")

    # Adjust spacing to ensure full visibility
    plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.1)

    # Show plot
    plt.show()

if __name__ == "__main__":
    file_path = "data.csv"
    df = load_data(file_path)
    
    if df is not None:
        df_clean = clean_dataset(df)

        # Perform Statistical Analysis and Print Results 📊
        print("\n🔍 Running Statistical Analysis...\n")
        stats_results = perform_statistical_analysis(df_clean)

# Generate Histogram for NTD Case Distribution
def plot_histogram(df, column="FactValueNumeric"):
    """
    Creates a histogram to visualize the distribution of NTD cases.

    Parameters:
    - df: pd.DataFrame, cleaned dataset.
    - column: str, the column containing NTD case counts.

    Returns:
    - Displays a histogram.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True, color="royalblue")

    plt.title("Distribution of NTD Cases", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("NTD Case Counts", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

# Improve annotation positioning in the line chart

def plot_trends_with_improved_annotations(df, column="FactValueNumeric", time_col="Period"):
    """
    Enhances trend visualization by improving key event annotations.

    Parameters:
    - df: pd.DataFrame, cleaned dataset.
    - column: str, the column containing NTD case counts.
    - time_col: str, the column containing time information.

    Returns:
    - Displays a line chart with improved annotations.
    """
    # Ensure period is numeric and sorted
    df.loc[:, time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.sort_values(by=time_col)

    # Aggregate yearly data
    time_trends = df.groupby(time_col)[column].sum()

    # Key events impacting NTD progress
    events = {
        2012: "London Declaration on NTDs",
        2015: "Global NTD Funding Boost",
        2017: "Ghana Eliminates Trachoma",
        2020: "COVID-19 Disruptions",
        2021: "WHO NTD Roadmap 2030"
    }

    # Plot the line chart
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(x=time_trends.index, y=time_trends.values, marker="o", color="darkred", linewidth=2.5)

    # Add vertical event markers and improve annotations
    for year, event in events.items():
        if year in time_trends.index:
            plt.axvline(x=year, color="gray", linestyle="--", alpha=0.7)  # Vertical line for the event
            
            # Adjust annotation positioning dynamically based on trend values
            y_position = time_trends.loc[year] * 1.02  # Slightly above the point
            plt.annotate(event, 
                         xy=(year, time_trends.loc[year]), 
                         xytext=(year, y_position), 
                         fontsize=10, color="black", 
                         ha="center", arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Title and labels
    plt.title("Trends in NTD Cases Over Time (2010-2021) with Key Events", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Total NTD Cases", fontsize=14)
    plt.xticks(time_trends.index, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

# Apply visualizations on the cleaned dataset
visualize_top_countries(df_clean)
plot_histogram(df_clean)
plot_trends_with_improved_annotations(df_clean)
