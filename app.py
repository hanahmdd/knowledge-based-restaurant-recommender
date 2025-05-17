import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import os
import time
from recommendation_engine import RestaurantRecommender

# Set page configuration
st.set_page_config(
    page_title="Zomato Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to load cached recommender system
@st.cache_resource
def load_recommender():
    with st.spinner("Loading restaurant data... This may take a moment."):
        recommender = RestaurantRecommender()
    return recommender


def display_map(restaurants_df):
    """Display restaurants on an interactive map"""
    if restaurants_df.empty:
        return None

    # Check if required coordinates exist
    if 'Latitude' not in restaurants_df.columns or 'Longitude' not in restaurants_df.columns:
        return None

    # Filter out rows with missing coordinates
    map_df = restaurants_df.dropna(subset=['Latitude', 'Longitude'])

    if map_df.empty:
        return None

    # Create a map centered at the mean coordinates
    center_lat = map_df['Latitude'].mean()
    center_lng = map_df['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)

    # Add markers for each restaurant
    for _, row in map_df.iterrows():
        # Prepare popup text with error handling
        name = row.get('Restaurant Name', 'Unknown').title()
        cuisines = row.get('Cuisines', 'Unknown')
        rating = row.get('Aggregate rating', 'N/A')
        votes = row.get('Votes', 0)
        currency = row.get('Currency', '')
        cost = row.get('Average Cost for two', 'N/A')

        popup_text = f"""
        <b>{name}</b><br>
        Cuisine: {cuisines}<br>
        Rating: {rating}/5 ({votes} votes)<br>
        Cost: {currency} {cost} for two<br>
        """

        # Add marker with popup
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=name,
            icon=folium.Icon(icon="cutlery", prefix="fa")
        ).add_to(m)

    return m


def display_restaurant_card(restaurant):
    """Display a restaurant as a card"""
    # Create a container for the card
    card = st.container()

    with card:
        col1, col2 = st.columns([3, 1])

        with col1:
            # Restaurant name and basic info
            name = restaurant.get('Restaurant Name', 'Unknown Restaurant')
            st.subheader(name.title() if isinstance(name, str) else str(name))

            cuisines = restaurant.get('Cuisines', 'Not specified')
            st.caption(f"Cuisines: {cuisines}")

            # Rating with colored badge
            rating = restaurant.get('Aggregate rating', 0)
            # Convert rating to float if it's not already
            if isinstance(rating, str):
                try:
                    rating = float(rating)
                except:
                    rating = 0

            rating_color = "green" if rating >= 4 else "orange" if rating >= 3 else "red"
            st.markdown(
                f"""<span style="background-color:{rating_color}; color:white; padding:3px 8px; 
                border-radius:10px; font-weight:bold;">{rating}/5</span> 
                <span style="color:gray;">({int(restaurant.get('Votes', 0))} votes)</span>""",
                unsafe_allow_html=True
            )

            # Cost and location
            currency = restaurant.get('Currency', '')
            cost = restaurant.get('Average Cost for two', 'N/A')
            st.markdown(f"**Cost:** {currency} {cost} for two")

            locality = restaurant.get('Locality', 'Unknown')
            if isinstance(locality, str):
                locality = locality.title()
            st.markdown(f"**Location:** {locality}")

            # Features as badges
            features = []
            if restaurant.get('Has Online delivery') in [True, 'Yes', 'yes', 1]:
                features.append("🛵 Delivery")
            if restaurant.get('Has Table booking') in [True, 'Yes', 'yes', 1]:
                features.append("🪑 Table Booking")

            if features:
                st.markdown(" | ".join(features))

        with col2:
            # Map link if coordinates are available
            lat = restaurant.get('Latitude')
            lng = restaurant.get('Longitude')
            if pd.notna(lat) and pd.notna(lng):
                map_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"
                st.markdown(f"[📍 View on Map]({map_url})")

    # Add a divider
    st.markdown("---")


def main():
    # Title and introduction
    st.title("🍽️ Zomato Restaurant Recommender")
    st.markdown("""
    Find the perfect restaurant based on your preferences - cuisine, budget, and location.
    """)

    # Debug mode toggle in the sidebar
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", False)

    # Load the recommender system
    try:
        recommender = load_recommender()

        if debug_mode:
            st.sidebar.success(f"Loaded recommender with {len(recommender.data)} restaurants")
            st.sidebar.write(f"Available columns: {recommender.data.columns.tolist()}")
    except Exception as e:
        st.error(f"Error loading recommender: {str(e)}")
        st.stop()

    # Sidebar for user preferences
    st.sidebar.header("Your Preferences")

    # Cuisine selection (multi-select)
    cuisine_options = ["All"] + recommender.all_cuisines

    if debug_mode:
        st.sidebar.write(f"Available cuisines: {len(recommender.all_cuisines)}")

    selected_cuisines = st.sidebar.multiselect(
        "Select cuisines 🍕",
        options=cuisine_options,
        default=["All"]
    )

    # Handle "All" selection
    if "All" in selected_cuisines:
        cuisines_filter = None  # No cuisine filter
    else:
        cuisines_filter = selected_cuisines

    # Budget range selection
    budget_options = {
        "Any Budget": None,
        "Low Budget": 1,
        "Medium-Low Budget": 2,
        "Medium-High Budget": 3,
        "High Budget": 4
    }

    selected_budget = st.sidebar.selectbox(
        "Select budget range 💰",
        options=list(budget_options.keys()),
        index=0
    )
    budget_filter = budget_options[selected_budget]

    # Location selection
    location_options = ["Any Location"] + recommender.all_locations

    if debug_mode:
        st.sidebar.write(f"Available locations: {len(recommender.all_locations)}")

    selected_location = st.sidebar.selectbox(
        "Select location 📍",
        options=location_options,
        index=0
    )

    location_filter = None if selected_location == "Any Location" else selected_location

    # Additional filters
    st.sidebar.header("Additional Filters")

    delivery_filter = st.sidebar.checkbox("Online Delivery Available 🛵")
    booking_filter = st.sidebar.checkbox("Table Booking Available 🪑")

    min_rating = st.sidebar.slider(
        "Minimum Rating ⭐",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5
    )

    num_results = st.sidebar.number_input(
        "Number of Results",
        min_value=1,
        max_value=50,
        value=10
    )

    # Search button
    search_button = st.sidebar.button("Find Restaurants 🔍")

    # Display dataset sample in debug mode
    if debug_mode:
        with st.expander("Dataset Sample"):
            st.dataframe(recommender.data.head())

            # Display some stats about crucial columns
            st.subheader("Dataset Statistics")

            # Check Cuisine List column
            if 'Cuisine List' in recommender.data.columns:
                first_cuisine = recommender.data['Cuisine List'].iloc[0] if not recommender.data.empty else None
                st.write(f"First cuisine list item type: {type(first_cuisine)}")
                st.write(f"First cuisine list value: {first_cuisine}")

            # Count rows with delivery option
            if 'Has Online delivery' in recommender.data.columns:
                delivery_count = recommender.data['Has Online delivery'].sum() if recommender.data[
                                                                                      'Has Online delivery'].dtype == bool else \
                recommender.data['Has Online delivery'].isin([True, 'Yes', 'yes', 1]).sum()
                st.write(f"Restaurants with delivery: {delivery_count}")

            # Count rows with table booking
            if 'Has Table booking' in recommender.data.columns:
                booking_count = recommender.data['Has Table booking'].sum() if recommender.data[
                                                                                   'Has Table booking'].dtype == bool else \
                recommender.data['Has Table booking'].isin([True, 'Yes', 'yes', 1]).sum()
                st.write(f"Restaurants with table booking: {booking_count}")

            # Rating distribution
            if 'Aggregate rating' in recommender.data.columns:
                st.write("Rating distribution:")
                st.write(recommender.data['Aggregate rating'].value_counts().sort_index())

    # Results display
    if search_button:
        with st.spinner("Finding the best restaurants for you..."):
            # Get recommendations based on user preferences
            recommendations = recommender.get_recommendations(
                cuisines=cuisines_filter,
                budget_range=budget_filter,
                location=location_filter,
                delivery_only=delivery_filter,
                table_booking=booking_filter,
                min_rating=min_rating,
                num_results=num_results
            )

            # Show results
            if recommendations.empty:
                st.warning("No restaurants found matching your criteria. Try adjusting your filters.")

                if debug_mode:
                    st.subheader("Debugging Filter Issues")

                    # Check cuisine filter
                    if cuisines_filter:
                        st.write(f"Selected cuisines: {cuisines_filter}")
                        if 'Cuisines' in recommender.data.columns:
                            cuisine_matches = []
                            for cuisine in cuisines_filter:
                                count = recommender.data['Cuisines'].str.contains(cuisine, case=False, na=False).sum()
                                cuisine_matches.append(f"{cuisine}: {count} restaurants")
                            st.write("Restaurants with selected cuisines:")
                            st.write("\n".join(cuisine_matches))

                    # Check location filter
                    if location_filter:
                        st.write(f"Selected location: {location_filter}")
                        if 'Locality' in recommender.data.columns:
                            location_count = recommender.data['Locality'].str.contains(location_filter, case=False,
                                                                                       na=False).sum()
                            st.write(f"Restaurants in selected location: {location_count}")

                    # Check additional filters
                    if delivery_filter:
                        if 'Has Online delivery' in recommender.data.columns:
                            delivery_count = recommender.data['Has Online delivery'].sum() if recommender.data[
                                                                                                  'Has Online delivery'].dtype == bool else \
                            recommender.data['Has Online delivery'].isin([True, 'Yes', 'yes', 1]).sum()
                            st.write(f"Restaurants with delivery: {delivery_count}")

                    if booking_filter:
                        if 'Has Table booking' in recommender.data.columns:
                            booking_count = recommender.data['Has Table booking'].sum() if recommender.data[
                                                                                               'Has Table booking'].dtype == bool else \
                            recommender.data['Has Table booking'].isin([True, 'Yes', 'yes', 1]).sum()
                            st.write(f"Restaurants with table booking: {booking_count}")

                    if min_rating > 0:
                        if 'Aggregate rating' in recommender.data.columns:
                            rating_count = (recommender.data['Aggregate rating'] >= min_rating).sum()
                            st.write(f"Restaurants with rating >= {min_rating}: {rating_count}")

                    st.write("Try removing some filters or making them less restrictive")
            else:
                st.success(f"Found {len(recommendations)} restaurants matching your criteria!")

                # Display results in tabs: List view and Map view
                tab1, tab2 = st.tabs(["📋 List View", "🗺️ Map View"])

                with tab1:
                    # Display recommendations as cards
                    for _, restaurant in recommendations.iterrows():
                        display_restaurant_card(restaurant)

                with tab2:
                    # Display map if coordinates are available
                    restaurant_map = display_map(recommendations)
                    if restaurant_map:
                        folium_static(restaurant_map, width=1000, height=500)
                    else:
                        st.info("Map view not available for these restaurants")

    # Disclaimer
    st.sidebar.markdown("---")
    st.sidebar.caption("Note: This recommender uses the Zomato dataset for demonstration purposes.")

    # User feedback and save to CSV
    st.sidebar.markdown("---")
    st.sidebar.subheader("How was your experience?")
    feedback = st.sidebar.radio(
        "Rate your satisfaction with recommendations:",
        options=["", "😞 Poor", "😐 Average", "🙂 Good", "😄 Excellent"],
        index=0
    )

    # Optional comment field
    feedback_comment = st.sidebar.text_area("Additional comments (optional):", height=100)

    if feedback and feedback != "":
        if st.sidebar.button("Submit Feedback"):
            # Prepare feedback data
            feedback_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'rating': feedback,
                'comment': feedback_comment,
                'cuisine_filter': str(cuisines_filter),
                'budget_filter': str(budget_filter),
                'location_filter': str(location_filter),
                'delivery_filter': delivery_filter,
                'booking_filter': booking_filter,
                'min_rating_filter': min_rating
            }

            # Convert to DataFrame
            feedback_df = pd.DataFrame([feedback_data])

            # Append to CSV or create new one if it doesn't exist
            feedback_path = os.path.join('data', 'feedback.csv')

            if os.path.exists(feedback_path):
                feedback_df.to_csv(feedback_path, mode='a', header=False, index=False)
            else:
                feedback_df.to_csv(feedback_path, index=False)

            st.sidebar.success("Thank you for your feedback! Your response has been saved.")


if __name__ == "__main__":
    main()