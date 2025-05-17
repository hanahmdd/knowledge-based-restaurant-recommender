

import pandas as pd
import numpy as np
import os





def get_data_path(filename):
    """Get the correct path to data files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    return os.path.join(data_dir, filename)


def load_data():
    """Load the Zomato dataset from the data directory"""
    try:
        # Get paths to data files using relative paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')

        zomato_path = os.path.join(data_dir, "zomato.csv")
        country_path = os.path.join(data_dir, "Country-Code.xlsx")

        print(f"Looking for data in: {data_dir}")
        print(f"Files in data directory: {os.listdir(data_dir)}")



        # Verify files exist
        if not os.path.exists(zomato_path):
            raise FileNotFoundError(f"Zomato CSV not found at {zomato_path}")
        if not os.path.exists(country_path):
            raise FileNotFoundError(f"Country code Excel not found at {country_path}")

        # Load data with multiple encoding attempts
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(zomato_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            df = pd.read_csv(zomato_path, encoding='utf-8', errors='replace')
            print("Used error replacement for encoding")

        # Load country data
        country_df = pd.read_excel(country_path)
        print("Successfully loaded country codes")

        # Merge datasets
        merged_df = pd.merge(df, country_df, on='Country Code', how='left')
        print(f"Merged data shape: {merged_df.shape}")

        return merged_df

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return None

def clean_data(df):
    """
    Clean the Zomato dataset by removing duplicates, handling missing values,
    and normalizing categorical values.
    """
    if df is None:
        print("Error: No data to clean")
        return None

    print("Starting data cleaning...")

    # Make a copy to avoid warnings
    cleaned_df = df.copy()

    # Remove duplicates
    initial_rows = cleaned_df.shape[0]
    cleaned_df = cleaned_df.drop_duplicates(subset=['Restaurant ID'])
    print(f"Removed {initial_rows - cleaned_df.shape[0]} duplicate rows")

    # Handle missing values
    # Fill missing cuisines with 'Not Specified'
    if 'Cuisines' in cleaned_df.columns:
        cleaned_df['Cuisines'] = cleaned_df['Cuisines'].fillna('Not Specified')
        # Convert empty strings to NaN and then fill
        cleaned_df['Cuisines'] = cleaned_df['Cuisines'].replace('', 'Not Specified')

    # Drop rows where crucial columns are missing
    crucial_columns = ['Restaurant Name', 'Cuisines', 'Average Cost for two', 'Aggregate rating']
    # Make sure all crucial columns exist
    crucial_columns = [col for col in crucial_columns if col in cleaned_df.columns]

    if crucial_columns:  # Only proceed if there are crucial columns
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.dropna(subset=crucial_columns)
        print(f"Dropped {initial_rows - cleaned_df.shape[0]} rows with missing crucial data")

    # Normalize text columns - convert to lowercase
    text_columns = ['Restaurant Name', 'Cuisines', 'Locality', 'Locality Verbose']
    for col in text_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.lower()

    # Convert ratings and cost to numerical formats
    numeric_columns = ['Aggregate rating', 'Average Cost for two', 'Votes']
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            # Fill missing numerical values with appropriate defaults
            if col in ['Aggregate rating', 'Average Cost for two']:
                cleaned_df[col] = cleaned_df[col].fillna(0)
            if col == 'Votes':
                cleaned_df[col] = cleaned_df[col].fillna(0)

    # Convert boolean columns to proper boolean values
    bool_columns = ['Has Table booking', 'Has Online delivery', 'Is delivering now']
    for col in bool_columns:
        if col in cleaned_df.columns:
            # Convert 'Yes'/'No' to True/False if needed
            if cleaned_df[col].dtype == object:
                cleaned_df[col] = cleaned_df[col].map({'Yes': True, 'No': False})

    print("Data cleaning completed")
    return cleaned_df


def engineer_features(df):
    """
    Engineer features for the recommendation system:
    - Categorize cost into buckets
    - Extract primary cuisine
    - Map ratings to consistent scale
    """
    if df is None:
        print("Error: No data for feature engineering")
        return None

    print("Starting feature engineering...")

    # Make a copy to avoid warnings
    featured_df = df.copy()

    # Check if required columns exist
    required_columns = ['Average Cost for two', 'Cuisines']
    missing_columns = [col for col in required_columns if col not in featured_df.columns]

    if missing_columns:
        print(f"Error: Missing required columns for feature engineering: {missing_columns}")
        return None

    # 1. Categorize cost into buckets (low, medium, high)
    # Calculate quartiles for cost categorization
    q1 = featured_df['Average Cost for two'].quantile(0.25)
    q2 = featured_df['Average Cost for two'].quantile(0.5)
    q3 = featured_df['Average Cost for two'].quantile(0.75)

    def categorize_cost(cost):
        if cost <= q1:
            return 'low'
        elif cost <= q3:
            return 'medium'
        else:
            return 'high'

    featured_df['Cost Category'] = featured_df['Average Cost for two'].apply(categorize_cost)

    # Create a more specific budget range for filtering
    def get_budget_range(cost):
        if cost <= q1:
            return 1  # Low
        elif cost <= q2:
            return 2  # Medium-Low
        elif cost <= q3:
            return 3  # Medium-High
        else:
            return 4  # High

    featured_df['Budget Range'] = featured_df['Average Cost for two'].apply(get_budget_range)

    # 2. Extract primary cuisine (first one listed)
    def extract_primary_cuisine(cuisines):
        if pd.isna(cuisines) or cuisines == 'Not Specified':
            return 'Not Specified'
        return cuisines.split(',')[0].strip()

    featured_df['Primary Cuisine'] = featured_df['Cuisines'].apply(extract_primary_cuisine)

    # 3. Create list of all cuisines for each restaurant
    featured_df['Cuisine List'] = featured_df['Cuisines'].apply(
        lambda x: [cuisine.strip() for cuisine in str(x).split(',')] if pd.notna(x) else []
    )

    # 4. Ensure rating is on a 0-5 scale (assuming it already is, but normalize to be sure)
    if 'Aggregate rating' in featured_df.columns:
        max_rating = featured_df['Aggregate rating'].max()
        if max_rating > 5:
            featured_df['Normalized Rating'] = featured_df['Aggregate rating'] * (5 / max_rating)
        else:
            featured_df['Normalized Rating'] = featured_df['Aggregate rating']

    # 5. Create a popularity score based on votes and rating
    # Normalize votes to 0-1 scale
    if 'Votes' in featured_df.columns and 'Normalized Rating' in featured_df.columns:
        max_votes = featured_df['Votes'].max()
        if max_votes > 0:
            featured_df['Normalized Votes'] = featured_df['Votes'] / max_votes
        else:
            featured_df['Normalized Votes'] = 0

        # Weighted score combining rating and votes
        featured_df['Popularity Score'] = (0.7 * featured_df['Normalized Rating'] / 5) + (
                0.3 * featured_df['Normalized Votes'])
    else:
        # Default popularity score if we don't have the required columns
        featured_df['Popularity Score'] = 0.5  # Neutral score

    print("Feature engineering completed")
    return featured_df


def prepare_data():
    """
    Main function to load, clean, and prepare the Zomato dataset for the recommendation engine
    """
    # Step 1: Load data
    raw_df = load_data()

    if raw_df is None:
        print("Error: Failed to load data")
        return None

    # Step 2: Clean data
    cleaned_df = clean_data(raw_df)

    if cleaned_df is None:
        print("Error: Failed to clean data")
        return None

    # Step 3: Engineer features
    processed_df = engineer_features(cleaned_df)

    if processed_df is not None:
        # Save the processed data for future use
        try:
            processed_path = os.path.join('data', 'processed_zomato.csv')
            processed_df.to_csv(processed_path, index=False)
            print(f"Processed data saved to {processed_path}")
        except Exception as e:
            print(f"Warning: Failed to save processed data: {e}")

    # Return the processed dataframe
    return processed_df


if __name__ == "__main__":
    # Execute the data preparation pipeline when run as a script
    processed_data = prepare_data()

    if processed_data is not None:
        print(f"Data preparation complete. Final dataset shape: {processed_data.shape}")
    else:
        print("Data preparation failed.")