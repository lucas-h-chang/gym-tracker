import streamlit as st
import sqlite3
import pandas as pd
import altair as alt
from datetime import datetime 
import pytz

california_tz = pytz.timezone('America/Los_Angeles')
current_time = datetime.now(california_tz)

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
st.write("More advanced predictions coming soon")
# 2. Connect to your local database file
conn = sqlite3.connect("gym_history.db")
df = pd.read_sql_query("SELECT * FROM capacity_log", conn)


##
# --- PREDICTION SECTION ---
##
st.header("Weekly Patterns")



df['timestamp'] = pd.to_datetime(df['timestamp'])
df['day_of_week'] = df['timestamp'].dt.day_name()


df['hour_label'] = df['timestamp'].dt.round('15min').dt.strftime('%I:%M %p').str.lstrip('0')

# Create a numeric helper to keep the times in the correct order (7.25, 7.5, etc.)
df['hour_numeric'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60

# 2. Select Day
target_day = st.selectbox("Select a day:", 
                          ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

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

# This snapstimes like 2:00:01 and 2:00:45 into the same bucket (14.00)
day_data['hour_numeric'] = day_data['hour_numeric'].round(2)

# Update your groupby to ONLY use 'hour'
avg_data = day_data.groupby(['hour_numeric', 'hour_label'])['percent_full'].mean().reset_index()

# Sort the data so the chart goes in proper chronological order
avg_data = avg_data.sort_values('hour_numeric')

#hour labels
hour_ticks = [label for label in avg_data['hour_label'].unique() if ':00' in str(label)]


line = alt.Chart(avg_data).mark_line(color='#FDB927', strokeWidth=3).encode(
    
    x=alt.X('hour_numeric:Q', 
        title='Time of Day',
        axis=alt.Axis(
            values=avg_data['hour_numeric'].unique().tolist(), # Use numeric values for placement
            labelExpr="datum.value % 1 === 0 ? datum.value + ':00' : ''" # Optional: cleaner labels
        )
    ),
    y=alt.Y('percent_full:Q', title='Average Capacity (%)', scale=alt.Scale(domain=[0,100])),
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
st.subheader("Raw Data Log")
st.dataframe(df)


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





