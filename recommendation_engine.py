import pandas as pd
import numpy as np
import os
from data_preprocessing import prepare_data


class RestaurantRecommender:
    def __init__(self, df=None):
        """
        Initialize the recommender with a preprocessed dataframe
        or load and prepare the data if none is provided
        """
        try:
            if df is not None:
                self.data = df
            else:
                processed_path = os.path.join('data', 'processed_zomato.csv')
                if os.path.exists(processed_path):
                    print(f"Loading preprocessed data from {processed_path}")
                    self.data = pd.read_csv(processed_path)
                else:
                    print("Preprocessing data from scratch...")
                    self.data = prepare_data()

                    if self.data is None:
                        raise ValueError("Data preparation returned None")

            # Verify data was loaded properly
            if self.data is None:
                raise ValueError("No data loaded after initialization")

            # Ensure Cuisine List is properly formatted
            self._ensure_cuisine_list_format()

            # Extract all unique cuisines for the UI
            self.all_cuisines = self.extract_all_cuisines()

            # Extract all unique locations for the UI
            self.all_locations = self.extract_all_locations()

            print(f"Recommender initialized with {len(self.data)} restaurants")
            print(f"Available cuisines: {len(self.all_cuisines)}")
            print(f"Available locations: {len(self.all_locations)}")

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            if 'data' in os.listdir():
                print(f"Data directory contents: {os.listdir('data')}")
            else:
                print("Data directory not found")
            raise ValueError("Failed to initialize recommender. Please check data files and paths.")

    def _ensure_cuisine_list_format(self):
        """Make sure Cuisine List is properly formatted as a list object"""
        # Create Cuisine List from Cuisines if it doesn't exist
        if 'Cuisine List' not in self.data.columns and 'Cuisines' in self.data.columns:
            print("Creating 'Cuisine List' from 'Cuisines' column")
            self.data['Cuisine List'] = self.data['Cuisines'].apply(
                lambda x: [cuisine.strip() for cuisine in str(x).split(',')] if pd.notna(x) else []
            )

        # Convert string representation of lists back to actual lists
        elif 'Cuisine List' in self.data.columns:
            # Check the first non-null value to determine the type
            sample = self.data['Cuisine List'].dropna().iloc[0] if not self.data[
                'Cuisine List'].dropna().empty else None

            if isinstance(sample, str):
                print("Converting string cuisine lists to actual lists")
                self.data['Cuisine List'] = self.data['Cuisine List'].apply(
                    lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else
                    [c.strip() for c in str(x).split(',')] if isinstance(x, str) else []
                )

    def extract_all_cuisines(self):
        """Extract all unique cuisines from the dataset"""
        all_cuisines = set()

        # Try to get cuisines from Cuisine List first
        if 'Cuisine List' in self.data.columns:
            for cuisines in self.data['Cuisine List']:
                if isinstance(cuisines, list):
                    all_cuisines.update(cuisines)
                else:
                    # Skip if not a list
                    continue
        # Fall back to Cuisines column
        elif 'Cuisines' in self.data.columns:
            for cuisines_str in self.data['Cuisines'].dropna():
                all_cuisines.update([c.strip() for c in str(cuisines_str).split(',')])

        # Filter out empty or 'not specified' cuisines
        all_cuisines = [cuisine for cuisine in all_cuisines if cuisine and cuisine.lower() != 'not specified']
        return sorted(all_cuisines)

    def extract_all_locations(self):
        """Extract all unique localities from the dataset"""
        if 'Locality' in self.data.columns:
            locations = self.data['Locality'].dropna().unique().tolist()
            return sorted([loc for loc in locations if loc and loc.lower() != 'not specified'])
        return []

    def filter_restaurants(self, cuisines=None, budget_range=None, location=None, delivery_only=False,
                           table_booking=False, min_rating=0):
        """Filter restaurants based on user preferences"""
        # Start with a clean copy of the data
        filtered_df = self.data.copy()

        # Debug info
        initial_count = len(filtered_df)
        print(f"Starting filter with {initial_count} restaurants")

        # Apply cuisine filter
        if cuisines and len(cuisines) > 0:
            print(f"Filtering by cuisines: {cuisines}")
            # Handle filtering based on Cuisine List column (preferred)
            if 'Cuisine List' in filtered_df.columns:
                # Ensure Cuisine List contains list objects
                if isinstance(filtered_df['Cuisine List'].iloc[0], str):
                    filtered_df['Cuisine List'] = filtered_df['Cuisine List'].apply(
                        lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else
                        [c.strip() for c in str(x).split(',')] if isinstance(x, str) else []
                    )

                # More lenient matching - check if any selected cuisine appears in the restaurant's cuisine list
                filtered_df = filtered_df[filtered_df['Cuisine List'].apply(
                    lambda cuisine_list: any(cuisine.lower() in [c.lower() for c in cuisine_list]
                                             if isinstance(cuisine_list, list) else False
                                             for cuisine in cuisines)
                )]
            # Fall back to Cuisines string column
            elif 'Cuisines' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Cuisines'].apply(
                    lambda cuisines_str: any(cuisine.lower() in str(cuisines_str).lower()
                                             for cuisine in cuisines)
                    if pd.notna(cuisines_str) else False
                )]

            print(f"After cuisine filter: {len(filtered_df)} restaurants")

        # Apply budget filter
        if budget_range is not None and 'Budget Range' in filtered_df.columns:
            print(f"Filtering by budget: {budget_range}")
            # Handle both list range and single value
            if isinstance(budget_range, list) and len(budget_range) == 2:
                min_budget, max_budget = budget_range
                filtered_df = filtered_df[
                    (filtered_df['Budget Range'] >= min_budget) &
                    (filtered_df['Budget Range'] <= max_budget)
                    ]
            else:
                # More lenient matching for budget
                filtered_df = filtered_df[filtered_df['Budget Range'] == budget_range]

            print(f"After budget filter: {len(filtered_df)} restaurants")

        # Apply location filter
        if location and location.strip() and 'Locality' in filtered_df.columns:
            print(f"Filtering by location: {location}")
            # More lenient matching - partial case-insensitive match
            filtered_df = filtered_df[
                filtered_df['Locality'].str.contains(location.lower(), case=False, na=False)
            ]
            print(f"After location filter: {len(filtered_df)} restaurants")

        # Apply delivery filter
        if delivery_only and 'Has Online delivery' in filtered_df.columns:
            print("Filtering for delivery only")
            # Handle different formats of boolean values
            if filtered_df['Has Online delivery'].dtype == bool:
                filtered_df = filtered_df[filtered_df['Has Online delivery'] == True]
            else:
                # Handle Yes/No strings
                filtered_df = filtered_df[filtered_df['Has Online delivery'].isin([True, 'Yes', 'yes', 1])]

            print(f"After delivery filter: {len(filtered_df)} restaurants")

        # Apply table booking filter
        if table_booking and 'Has Table booking' in filtered_df.columns:
            print("Filtering for table booking")
            # Handle different formats of boolean values
            if filtered_df['Has Table booking'].dtype == bool:
                filtered_df = filtered_df[filtered_df['Has Table booking'] == True]
            else:
                # Handle Yes/No strings
                filtered_df = filtered_df[filtered_df['Has Table booking'].isin([True, 'Yes', 'yes', 1])]

            print(f"After booking filter: {len(filtered_df)} restaurants")

        # Apply rating filter
        if min_rating > 0 and 'Aggregate rating' in filtered_df.columns:
            print(f"Filtering by minimum rating: {min_rating}")
            filtered_df = filtered_df[filtered_df['Aggregate rating'] >= min_rating]
            print(f"After rating filter: {len(filtered_df)} restaurants")

        # Final count
        print(f"Final filter result: {len(filtered_df)} restaurants")
        return filtered_df

    def rank_restaurants(self, filtered_df, ranking_weights=None):
        """Rank filtered restaurants by score"""
        if filtered_df.empty:
            return filtered_df

        if not ranking_weights:
            ranking_weights = {
                'rating': 0.6,
                'votes': 0.2,
                'popularity': 0.2
            }

        ranked_df = filtered_df.copy()

        # Ensure all required columns exist
        required_columns = ['Aggregate rating', 'Votes', 'Popularity Score']
        for col in required_columns:
            if col not in ranked_df.columns:
                if col == 'Aggregate rating':
                    ranked_df[col] = 0.0
                elif col == 'Votes':
                    ranked_df[col] = 0
                elif col == 'Popularity Score':
                    ranked_df[col] = 0.0

        # Convert string ratings to float if needed
        if ranked_df['Aggregate rating'].dtype == object:
            ranked_df['Aggregate rating'] = pd.to_numeric(ranked_df['Aggregate rating'], errors='coerce').fillna(0)

        # Convert string votes to int if needed
        if ranked_df['Votes'].dtype == object:
            ranked_df['Votes'] = pd.to_numeric(ranked_df['Votes'], errors='coerce').fillna(0)

        max_rating = 5.0
        ranked_df['Rating Score'] = ranked_df['Aggregate rating'] / max_rating

        max_votes = ranked_df['Votes'].max()
        if max_votes > 0:
            ranked_df['Vote Score'] = ranked_df['Votes'] / max_votes
        else:
            ranked_df['Vote Score'] = 0

        ranked_df['Final Score'] = (
                ranking_weights['rating'] * ranked_df['Rating Score'] +
                ranking_weights['votes'] * ranked_df['Vote Score'] +
                ranking_weights['popularity'] * ranked_df.get('Popularity Score', 0)
        )

        ranked_df = ranked_df.sort_values('Final Score', ascending=False)

        return ranked_df

    def get_recommendations(self, cuisines=None, budget_range=None, location=None, delivery_only=False,
                            table_booking=False, min_rating=0, num_results=10, ranking_weights=None):
        """Get restaurant recommendations based on user preferences"""
        # Print debug info about filters
        print(f"\nGetting recommendations with filters:")
        print(f"- Cuisines: {cuisines}")
        print(f"- Budget Range: {budget_range}")
        print(f"- Location: {location}")
        print(f"- Delivery Only: {delivery_only}")
        print(f"- Table Booking: {table_booking}")
        print(f"- Min Rating: {min_rating}")

        # Apply filters to get matching restaurants
        filtered_restaurants = self.filter_restaurants(
            cuisines, budget_range, location, delivery_only, table_booking, min_rating
        )

        if filtered_restaurants.empty:
            print("No restaurants found matching criteria")
            return pd.DataFrame()

        # Rank the filtered restaurants
        ranked_restaurants = self.rank_restaurants(filtered_restaurants, ranking_weights)

        # Get the top recommendations
        top_recommendations = ranked_restaurants.head(num_results).copy()
        print(f"Returning {len(top_recommendations)} recommendations")

        return top_recommendations
