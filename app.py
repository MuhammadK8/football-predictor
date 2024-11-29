import streamlit as st
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from streamlit_lottie import st_lottie  # Ensure this is properly imported

# Load the model from the 'model' folder
model_path = os.path.join(os.path.dirname(__file__), 'model', 'rf_rolling_model.pkl')
model = joblib.load(model_path)

# Load the opponent_to_code mapping from the 'data' folder
mapping_path = os.path.join(os.path.dirname(__file__), 'data', 'opponent_to_code.json')
with open(mapping_path, 'r') as f:
    opponent_to_code = json.load(f)

# Load match data from the 'data' folder and precompute rolling averages
@st.cache_data
def load_data():
    matches_path = os.path.join(os.path.dirname(__file__), 'data', 'pl_matches.csv')
    matches = pd.read_csv(matches_path)
    
    # Ensure opp_code in matches is consistent with opponent_to_code
    matches['opp_code'] = matches['opponent'].map(opponent_to_code)
    
    # Rolling averages
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
    new_cols = [f"{c}_rolling" for c in cols]
    
    def rolling_averages(group, cols, new_cols):
        group = group.sort_values("date")
        rolling_stats = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset=new_cols)
        return group
    
    matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])
    
    return matches, matches_rolling, new_cols

matches, matches_rolling, new_cols = load_data()

# Load Lottie animations from the 'static' folder
def load_lottie(filename):
    filepath = os.path.join(os.path.dirname(__file__), 'static', filename)
    with open(filepath, "r") as f:
        return json.load(f)

intro_lottie = load_lottie("intro - anim.json")  # Load the title animation
win_lottie = load_lottie("win - anim.json")  # Load the win animation

# Streamlit app
st_lottie(intro_lottie, height=300, key="intro")  # Display the Lottie animation
st.markdown(
    """
    <h1 style='text-align: center;'>Football Match Winner Predictor</h1>
    """,
    unsafe_allow_html=True
)

# Dropdowns for team selection
team_options = sorted(opponent_to_code.keys())
home_team = st.selectbox("Select Home Team", options=team_options)
away_team = st.selectbox("Select Away Team", options=team_options)

# Date and hour picker
match_date = st.date_input("Select Match Date")
match_hour = st.slider("Select Match Hour", min_value=0, max_value=23, value=20)

# Predict Button
if not st.button("Predict"): 
    st.write("Set your match details to see the prediction results here!")
else:
    # Check if home team and away team are the same
    if home_team == away_team:
        st.error(f"Oops! {home_team} can't play against themselves. Even in practice, they'd split into two squads!")
        st.stop()  # Halts execution if teams are the same

    try:
        # Convert inputs
        venue_code = 0  # Always home for the home team
        opp_code = opponent_to_code[away_team]
        day_code = match_date.weekday()
        
        # Retrieve rolling averages for the home team
        home_team_rolling = matches_rolling[matches_rolling['team'] == home_team]
        filtered_data = home_team_rolling[home_team_rolling['date'] < str(match_date)]
        
        if filtered_data.empty:
            # Fallback mechanism if no rolling data is available
            rolling_features = {col: 0 for col in new_cols}  # Default values
        else:
            latest_rolling = filtered_data.iloc[-1]
            rolling_features = {col: latest_rolling[col] for col in new_cols}
        
        # Prepare input
        input_data = pd.DataFrame([{**{
            "venue_code": venue_code,
            "opp_code": opp_code,
            "hour": match_hour,
            "day_code": day_code
        }, **rolling_features}])
        
        # Predict
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data).tolist()
        
        # Determine the result text
        if prediction[0] == 1:
            result = f"Home Win for {home_team}"
            winning_team = home_team
        else:
            result = f"Draw/Win for {away_team}"
            winning_team = away_team
        
        # Display results
        st.write(f"Prediction for **{home_team} (Home)** vs **{away_team} (Away)**")
        st.write(f"Predicted Result: **{result}**")
        st.write(f"Win Probability for {home_team} (Home): **{probability[0][1]*100:.2f}%**")
        
        # Show Lottie animation for winning team
        if prediction[0] == 1:  # Only show for a win
            st.write(f"**Woohoo! {winning_team}!**")
            st_lottie(win_lottie, height=200, key="win-animation")
    except Exception as e:
        st.error(f"Error: {e}")
