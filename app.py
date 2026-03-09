import streamlit as st
import sqlite3
import pandas as pd
import altair as alt
from datetime import datetime
import pytz
import pickle
import json
import os
import torch
from train import GymMLP, engineer_features

california_tz = pytz.timezone('America/Los_Angeles')
current_time = datetime.now(california_tz)


# Cache model loading so Streamlit doesn't reload models on every user interaction.
# @st.cache_resource keeps the models in memory for the lifetime of the server process.
@st.cache_resource
def load_models():
    with open("models/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/model_config.pkl", "rb") as f:
        config = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    model = GymMLP(config['n_features'])
    model.load_state_dict(torch.load("models/pytorch_model.pt", weights_only=True))
    model.eval()
    return rf, scaler, model, feature_names

current_hour = current_time.hour
current_day = current_time.strftime('%A')

# 2. Check the RSF hours using the California time
is_open = False
##
# RSF OPEN/CLOSE STATUS
##
if current_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
    is_open = 7 <= current_hour < 23
elif current_day == 'Saturday':
    is_open = 8 <= current_hour < 18
else: # Sunday
    is_open = 8 <= current_hour < 23

if is_open:
    st.success("**RSF is currently OPEN**")
else:
    st.error("**RSF is currently CLOSED**")

##
# TITLE
##
st.title("Berkeley RSF Weight Room Capacity Tracker")
st.write("Historical capacity data and predicted capacity at every 15-minute interval.")
st.write("Historical capacity data and predicted capacity.")

##
# PREDICT CAPACITY SECTION
##
st.header("Predict Capacity")

if not os.path.exists("models/rf_model.pkl"):
    st.info("Models not yet trained. Run `python3 train.py` locally or trigger the GitHub Action.")
else:
    rf, scaler, mlp_model, feature_names = load_models()

    col1, col2 = st.columns(2)
    with col1:
        pred_date = st.date_input("Date", value=current_time.date())

    # Determine open hours for the selected day
    pred_day = pd.Timestamp(pred_date).day_name()
    if pred_day == 'Saturday':
        open_hour, close_hour = 8, 18
    elif pred_day == 'Sunday':
        open_hour, close_hour = 8, 23
    else:
        open_hour, close_hour = 7, 23

    # Build list of valid 15-min slots within open hours
    from datetime import time as dtime
    time_slots = [
        dtime(h, m)
        for h in range(open_hour, close_hour)
        for m in (0, 15, 30, 45)
    ]
    time_labels = [t.strftime('%I:%M %p').lstrip('0') for t in time_slots]

    # If the user previously picked a time and it's still valid for this day, keep it.
    # Otherwise default to the nearest slot to current time.
    prev_label = st.session_state.get('pred_time_label')
    if prev_label in time_labels:
        default_idx = time_labels.index(prev_label)
    else:
        current_minutes = current_time.hour * 60 + current_time.minute
        default_idx = min(
            range(len(time_slots)),
            key=lambda i: abs(time_slots[i].hour * 60 + time_slots[i].minute - current_minutes)
        )

    with col2:
        selected_label = st.selectbox("Time", time_labels, index=default_idx, key='pred_time_label')
    pred_time = time_slots[time_labels.index(selected_label)]

    pred_timestamp = pd.Timestamp(f"{pred_date} {pred_time}")
    pred_df = pd.DataFrame({
        'timestamp': [pred_timestamp],
        'people_count': [100],
        'percent_full': [66.7]
    })
    X, _ = engineer_features(pred_df)

    rf_pred = rf.predict(X)[0]
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        mlp_pred = mlp_model(torch.tensor(X_scaled, dtype=torch.float32)).item()

    rf_display  = min(rf_pred,  100)
    mlp_display = min(mlp_pred, 100)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Neural Network (MLP)", f"{mlp_display:.0f}%",
                  help="PyTorch 3-layer neural network trained on ~140k data points")
        if mlp_pred > 100:
            st.caption("At or over capacity")
    with col2:
        st.metric("Random Forest", f"{rf_display:.0f}%",
                  help="Ensemble of 100 decision trees (scikit-learn)")
        if rf_pred > 100:
            st.caption("At or over capacity")

    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        st.caption(
            f"Last trained: {metrics['trained_at'][:10]} on {metrics['training_rows']:,} rows — "
            f"MLP avg error: ±{metrics['mlp']['mae']:.1f}%, RF avg error: ±{metrics['rf']['mae']:.1f}%"
        )

    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    importance_chart = alt.Chart(importances_df).mark_bar(color='#FDB927').encode(
        x=alt.X('Importance:Q', title='Importance Score'),
        y=alt.Y('Feature:N', sort='-x', title=None),
        tooltip=['Feature', alt.Tooltip('Importance:Q', format='.3f')]
    ).properties(title='What Drives Predictions (Random Forest feature importances)', height=320)

    st.altair_chart(importance_chart, use_container_width=True)


@st.cache_data(ttl=86400)  # cache for 1 day; clears on app restart anyway
def load_data():
    conn = sqlite3.connect("gym_history.db")
    return pd.read_sql_query("SELECT * FROM capacity_log", conn)

df = load_data()


##
# WEEKLY PATTERNS SECTION
##
st.header("Weekly Patterns")



df['timestamp'] = pd.to_datetime(df['timestamp'])
df['day_of_week'] = df['timestamp'].dt.day_name()

df['hour_label'] = df['timestamp'].dt.round('15min').dt.strftime('%I:%M %p').str.lstrip('0')

# Create a numeric helper to keep the times in the correct order (7.25, 7.5, etc.)
df['hour_numeric'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60

col_day, col_right = st.columns([2, 1])

with col_day:
    # Default to today so the chart is immediately relevant
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    target_day = st.selectbox("Select a day:", days, index=days.index(current_day))

with col_right:
    time_range = st.selectbox("Data range:", [
        "Last week", "Last month", "Last 6 months", "Last year", "All time"
    ])
    semester_only = st.checkbox("Semester only", value=True,
        help="Excludes summer, winter break, and spring break, so these averages reflect normal school weeks only. Nice.")

# Apply time range filter first
import numpy as np
from datetime import timedelta
cutoffs = {
    "Last week":     current_time - timedelta(days=7),
    "Last month":    current_time - timedelta(days=30),
    "Last 6 months": current_time - timedelta(days=182),
    "Last year":     current_time - timedelta(days=365),
    "All time":      None,
}
cutoff = cutoffs[time_range]
if cutoff is not None:
    cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
    df = df[df['timestamp'] >= cutoff.replace(tzinfo=None)]

# Then apply semester filter (stacks on top of time range)
if semester_only:
    month = df['timestamp'].dt.month
    day_of_month = df['timestamp'].dt.day
    date_only = df['timestamp'].dt.date

    is_summer       = month.isin([6, 7, 8])
    is_winter_break = ((month == 12) & (day_of_month >= 16)) | ((month == 1) & (day_of_month <= 12))

    # Spring break ranges (official Mon–Fri + surrounding Sat/Sun, matching train.py)
    spring_breaks = [
        ('2021-03-20', '2021-03-28'),
        ('2022-03-19', '2022-03-27'),
        ('2023-03-25', '2023-04-02'),
        ('2024-03-23', '2024-03-31'),
        ('2025-03-22', '2025-03-30'),
        ('2026-03-21', '2026-03-29'),
        ('2027-03-20', '2027-03-28'),
        ('2028-03-25', '2028-04-02'),
    ]
    is_spring_break = np.zeros(len(df), dtype=bool)
    for start, end in spring_breaks:
        is_spring_break |= (date_only >= pd.Timestamp(start).date()) & (date_only <= pd.Timestamp(end).date())

    df = df[~(is_summer | is_winter_break | is_spring_break)]

# 3. Filter for RSF Hours
if target_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
    start, end = 7, 23
elif target_day == 'Saturday':
    start, end = 8, 18
else: # Sunday
    start, end = 8, 23


day_data = df[(df['day_of_week'] == target_day) & 
              (df['hour_numeric'] >= start) & 
              (df['hour_numeric'] <= end)].copy()

# 2. THE FIX: Quantize to the nearest 15 minutes (0.25)
# This turns 15.0833 (3:05 PM) into 15.0 (3:00 PM)
day_data['hour_numeric'] = (day_data['hour_numeric'] * 4).round() / 4

# 3. Group by the newly 'binned' numbers
# We use .agg({'hour_label': 'first'}) to ensure only ONE label exists per bin
avg_data = day_data.groupby('hour_numeric').agg({
    'percent_full': 'mean',
    'hour_label': 'first'
}).reset_index()

# 4. Add a synthetic 0% row at closing time so every day's chart drops cleanly to 0.
# Without this, days like Saturday end at the last real data point (5:45 PM at ~50%)
# instead of visually closing at 6 PM. The closing-hour label is formatted to match.
close_hour = end  # end was set above based on the selected day
close_label = f"{close_hour % 12 or 12}:00 {'AM' if close_hour < 12 else 'PM'}"
closing_row = pd.DataFrame([{'hour_numeric': close_hour, 'percent_full': 0.0, 'hour_label': close_label}])
# Drop any real data at exactly closing time so the synthetic 0 is the only point there.
avg_data = avg_data[avg_data['hour_numeric'] < close_hour]
avg_data = pd.concat([avg_data, closing_row], ignore_index=True)

# 5. Sort strictly by the numeric value
avg_data = avg_data.sort_values('hour_numeric')
#hour labels
hour_ticks = [h for h in range(7, 24)] 

# 2. Update your chart encoding
line = alt.Chart(avg_data).mark_line(color='#FDB927', strokeWidth=3).encode(
    x=alt.X('hour_numeric:Q', 
        title='Time of Day',
        axis=alt.Axis(
            values=hour_ticks,
            # This logic: If 13, make it 1. If 0, make it 12. Add AM/PM based on 12.
            labelExpr="datum.value == 0 ? '12 AM' : datum.value == 12 ? '12 PM' : datum.value > 12 ? (datum.value - 12) + ' PM' : datum.value + ' AM'",
            grid=False,
            labelAngle=0 # Keeps labels horizontal and easy to read
        )
    ),
    y=alt.Y('percent_full:Q', 
        title='Average Capacity (%)', 
        scale=alt.Scale(domain=[0, 110]),
        axis=alt.Axis(grid=True)
    ),
    tooltip=[
        alt.Tooltip('hour_label', title='Time'),
        alt.Tooltip('percent_full', title='Avg Capacity', format='.1f')
    ]
)

# 5. The Points (Layer)
points = line.mark_point(color='#FDB927', size=60)

# 6. Combine AND THEN make interactive
# We use + to layer them, then call .interactive() on the result
final_predict_chart = (line + points).properties(
    height=300,
    padding={'right': 40}
).interactive(bind_y=False) # <--- This is the magic lock

st.altair_chart(final_predict_chart, use_container_width=True)




#
# Raw data log
#
# st.subheader("Raw Data Log")
# st.dataframe(df)


##
# Capacity graph
##
# st.subheader("Capacity Over Time")

# chart = alt.Chart(df).mark_line(
#     point=True, 
#     color='#5276A7', 
#     strokeWidth=3
# ).encode(
#     x=alt.X('timestamp:T', 
#         title='Time of Day',
#         axis=alt.Axis(
#             format='%I:%M %p', # Shows "01:00 PM" style
#             tickCount=5,       # Limits the number of time labels so they don't overlap
#             grid=False         # Removes vertical grid lines for a cleaner look
#         )
#     ), 
#     y=alt.Y('percent_full:Q', 
#         scale=alt.Scale(domain=[0, 100], clamp=True), 
#         title='Percent Full (%)',
#         axis=alt.Axis(tickCount=5) # Clean horizontal lines at 0, 25, 50, 75, 100
#     ),
#     tooltip=[
#         alt.Tooltip('timestamp:T', title='Time', format='%I:%M %p'),
#         alt.Tooltip('percent_full:Q', title='Capacity (%)')
#     ]
# ).properties(
#     height=400
# )
# # Keep the Y-axis locked while allowing X-axis zooming
# final_chart = chart.interactive(bind_y=False)

# st.altair_chart(final_chart, use_container_width=True)
