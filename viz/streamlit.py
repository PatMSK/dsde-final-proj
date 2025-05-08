import pandas as pd
import streamlit as st
import pydeck as pdk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MultipleLocator
import altair as alt  

st.header("Organizations in each district")
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv", usecols=["ticket_id", "organization", "district", "coords", "type", "timestamp"])
    df_stats = pd.read_csv("district_stats.csv")
    # Clean and prepare data
    df[['lon', 'lat']] = df['coords'].str.split(",", expand=True).astype(float)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except ValueError:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        except ValueError:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df['month_year'] = df['timestamp'].dt.to_period('M')
    df = df.dropna(subset=['lat', 'lon', 'timestamp'])
    return df, df_stats

# Load data
df,df_stats = load_data()

# districts
districts = df['district'].dropna().unique()
organizations = df['organization'].dropna().unique()
complaint_types = df['type'].dropna().unique()

COLOR_PALETTE = [
    [255, 99, 132, 200],   # Red
    [54, 162, 235, 200],   # Blue
    [255, 206, 86, 200],   # Yellow
    [75, 192, 192, 200],   # Green
    [153, 102, 255, 200], # Purple
    [255, 159, 64, 200],   # Orange
    [199, 199, 199, 200], # Gray
    [255, 0, 255, 200],     # Magenta
    [0, 255, 0, 200],       # Lime
    [0, 0, 0, 200],         # Black
]

# Create a color mapping dictionary for complaint types (define it before the function uses it)
complaint_color_mapping = {complaint_type: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, complaint_type in enumerate(complaint_types)}

# Sidebar controls for Map 1
st.sidebar.subheader("Controls for Map")
selected_district_map1 = st.sidebar.selectbox("เลือกเขต", sorted(districts), key="district_map1")
map1_filter_type = st.sidebar.radio("Filter Map by", ["Organization", "Complaint Type"], key="map1_filter_type")
filtered_df_map1 = df[df['district'] == selected_district_map1].copy()

#filtered df for map 1
orgs_map1 = filtered_df_map1['organization'].unique()
complaint_types_map1 = filtered_df_map1['type'].unique()
complaint_single_types = [ctype for ctype in complaint_types if len(str(ctype).split(',')) <= 1]

color_map_org_map1 = {
    org: COLOR_PALETTE[i % len(COLOR_PALETTE)]
    for i, org in enumerate(orgs_map1)
}
color_map_type_map1 = {
    ctype: COLOR_PALETTE[i % len(COLOR_PALETTE)]
    for i, ctype in enumerate(complaint_types_map1)
}

@st.cache_data
def each_district_scatter(filtered_df, color_column, color_mapping, selected_filter=None):
    if selected_filter and selected_filter != "All":
        filtered_df = filtered_df[filtered_df[color_column] == selected_filter]

    filtered_df["color"] = filtered_df[color_column].map(color_mapping)

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=filtered_df,
        get_position=['lon', 'lat'],
        get_fill_color="color",
        get_radius=10,
        pickable=True
    )
    view_state = pdk.ViewState(
        latitude=filtered_df['lat'].mean(),
        longitude=filtered_df['lon'].mean(),
        zoom=12
    )
    return layer, view_state

@st.cache_data
def each_district_heatmap(filtered_df, color_column, color_mapping, selected_filter=None):
    if selected_filter and selected_filter != "All":
        filtered_df = filtered_df[filtered_df[color_column] == selected_filter]

    filtered_df['color'] = filtered_df[color_column].map(color_mapping)

    layer = pdk.Layer(
        "HeatmapLayer",
        data=filtered_df,
        get_position=['lon', 'lat'],
        get_color="color",  # Use the 'color' column
        radius=50,
        intensity=1,
        threshold=0.05,
    )
    view_state = pdk.ViewState(
        latitude=filtered_df['lat'].mean(),
        longitude=filtered_df['lon'].mean(),
        zoom=12
    )
    return layer, view_state

@st.cache_data
def each_district_hexagon(filtered_df, selected_filter=None, filter_column=None):
    if selected_filter and selected_filter != "All" and filter_column:
        filtered_df = filtered_df[filtered_df[filter_column] == selected_filter]

    layer = pdk.Layer(
        "HexagonLayer",
        data=filtered_df,
        get_position=['lon', 'lat'],
        radius=100,
        color_aggregation="count",
        color_range=[[255, 255, 204], [199, 233, 180], [127, 205, 187], [65, 182, 196], [29, 145, 192], [34, 94, 168], [12, 44, 132]],
        pickable=True,
        extruded=True,
    )
    view_state = pdk.ViewState(
        latitude=filtered_df['lat'].mean(),
        longitude=filtered_df['lon'].mean(),
        zoom=12
    )
    return layer, view_state


if map1_filter_type == "Organization":
    org_counts_map1 = filtered_df_map1['organization'].value_counts().reset_index()
    org_counts_map1.columns = ['Organization', 'Count']
    st.subheader(f"Organization counts in {selected_district_map1}")
    st.dataframe(org_counts_map1)
    selected_org_map1 = st.sidebar.selectbox("Filter by Organization", ["All"] + list(orgs_map1), key="org_map1")
    color_column_map1 = "organization"
    color_mapping_map1 = color_map_org_map1
    tooltip_map1={"text": "องค์กร: {organization}\nเขต: {district}"}
elif map1_filter_type == "Complaint Type":
    complaint_counts_map1 = filtered_df_map1['type'].value_counts().reset_index()
    complaint_counts_map1.columns = ['Complaint Type', 'Count']
    st.subheader(f"Complaint Type counts in {selected_district_map1}")
    st.dataframe(complaint_counts_map1)
    selected_type_map1 = st.sidebar.selectbox("Filter by Complaint Type", ["All"] + list(complaint_single_types), key="type_map1")
    color_column_map1 = "type"
    color_mapping_map1 = color_map_type_map1
    tooltip_map1={"text": "ประเภทข้อร้องเรียน: {type}\nเขต: {district}"}
else:
    color_column_map1 = None
    color_mapping_map1 = None
    tooltip_map1 = {}

map_type = st.sidebar.radio(
    "Select Map Type",
    ["Scatter", "Heat", "Hexagon"],
    key="map1_type"
)

if map_type == "Scatter":
    if map1_filter_type == "Organization":
        layer_map1, view_state_map1 = each_district_scatter(filtered_df_map1, "organization", color_map_org_map1, selected_org_map1)
    elif map1_filter_type == "Complaint Type":
        layer_map1, view_state_map1 = each_district_scatter(filtered_df_map1, "type", color_map_type_map1, selected_type_map1)
    else:
        layer_map1, view_state_map1 = each_district_scatter(filtered_df_map1, "organization", color_map_org_map1) # Default to organization
elif map_type == "Heat":
    if map1_filter_type == "Organization":
        layer_map1, view_state_map1 = each_district_heatmap(filtered_df_map1, "organization", color_map_org_map1, selected_org_map1)
    elif map1_filter_type == "Complaint Type":
        layer_map1, view_state_map1 = each_district_heatmap(filtered_df_map1, "type", color_map_type_map1, selected_type_map1)
    else:
        layer_map1, view_state_map1 = each_district_heatmap(filtered_df_map1, "organization", color_map_org_map1) # Default to organization
    tooltip_map1 = {"text": "จำนวนจุด: {count}"}
elif map_type == "Hexagon":
    selected_filter_hexagon = None
    filter_column_hexagon = None
    if map1_filter_type == "Organization":
        selected_filter_hexagon = selected_org_map1
        filter_column_hexagon = "organization"
    elif map1_filter_type == "Complaint Type":
        selected_filter_hexagon = selected_type_map1
        filter_column_hexagon = "type"

    layer_map1, view_state_map1 = each_district_hexagon(filtered_df_map1, selected_filter_hexagon, filter_column_hexagon)
    tooltip_map1 = {"text": "จำนวนจุด: {count}"}

st.pydeck_chart(pdk.Deck(
    layers=[layer_map1],
    initial_view_state=view_state_map1,
    tooltip=tooltip_map1
))



# New Map for Complaint Type
st.header("Complaint Types in each district")

@st.cache_data
def complaint_type_scatter(filtered_df, selected_complaint_type=None, color_mapping=None): # Add color_mapping as argument
    # Filter by complaint type
    if selected_complaint_type and selected_complaint_type != "All":
        filtered_df = filtered_df[filtered_df['type'] == selected_complaint_type]

    # Map colors to the DataFrame based on complaint type
    if color_mapping:
        filtered_df['color'] = filtered_df['type'].map(color_mapping)
    else:
        # Fallback if color_mapping is not provided (though it should be)
        pass

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=filtered_df,
        get_position=['lon', 'lat'],
        get_fill_color="color",  # Use the 'color' column
        get_radius=10,
        pickable=True
    )
    view_state = pdk.ViewState(
        latitude=filtered_df['lat'].mean(),
        longitude=filtered_df['lon'].mean(),
        zoom=12
    )
    return layer, view_state

#district stats
st.header("District Statistics")

# font select
plt.rcParams['font.family'] = 'Tahoma'

# data
metrics = ["num_tickets", "avg_star", "avg_resolution_time", "resolution_rate"]

max_metric_values = {metric: df_stats[metric].max() * 1.1 for metric in metrics}
min_metric_values = {metric: df_stats[metric].min() * 0.9 for metric in metrics}

def plot_metric_bar_chart(df, metric, sort_order, num_districts, max_metric, min_metric):
    sorted_df = df.sort_values(by=metric, ascending=(sort_order == "Bottom"))
    subset = sorted_df.head(num_districts)

    cmap = cm.get_cmap('tab20', num_districts)
    colors = [cmap(i) for i in range(num_districts)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(subset["district"], subset[metric], color=colors)
    ax.set_title(f"{sort_order} {num_districts} District of {metric}", fontsize=14)
    ax.set_xlabel(metric)
    ax.set_ylabel("District")
    ax.invert_yaxis()

    ax.set_xlim(min_metric, max_metric)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center', ha='left', fontsize=9)

    st.pyplot(fig)

st.subheader("Metrics comparison by Bar graph")

st.sidebar.subheader("Controls for Overall Bar Graph")
sort_order = st.sidebar.radio("Select Rank", ["Top", "Bottom"])
num_districts = st.sidebar.slider("District Count", min_value=5, max_value=20, value=10)

for metric in metrics:
    st.subheader(metric)
    plot_metric_bar_chart(df_stats, metric, sort_order, num_districts, max_metric_values[metric], min_metric_values[metric])

# New Sidebar Section for Problem Breakdown
st.sidebar.subheader("Problem Breakdown in District")
selected_district_breakdown = st.sidebar.selectbox("เลือกเขต", sorted(districts), key="district_breakdown")
problem_type = st.sidebar.radio("View problems by", ["Organization", "Complaint Type"], key="problem_type")
num_items_to_show = st.sidebar.slider("Number of Items to Show", min_value=5, max_value=20, value=10, key="num_items_breakdown") # Added slider
filtered_df_breakdown = df[df['district'] == selected_district_breakdown].copy()

st.header("Problem Breakdown in Selected District")

if problem_type == "Organization":
    organization_counts = filtered_df_breakdown['organization'].value_counts().nlargest(num_items_to_show).reset_index() # Limit to top N
    organization_counts.columns = ['Organization', 'Count']
    st.subheader(f"Top {num_items_to_show} Organizations by Problem Count in {selected_district_breakdown}")
    st.dataframe(organization_counts)

    # Optional: Display a bar chart of organization counts
    fig_org, ax_org = plt.subplots()
    ax_org.bar(organization_counts['Organization'], organization_counts['Count'])
    ax_org.set_xlabel("Organization")
    ax_org.set_ylabel("Number of Problems")
    ax_org.set_title(f"Top {num_items_to_show} Organizations in {selected_district_breakdown}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_org)

elif problem_type == "Complaint Type":
    complaint_counts = filtered_df_breakdown['type'].value_counts().nlargest(num_items_to_show).reset_index() # Limit to top N
    complaint_counts.columns = ['Complaint Type', 'Count']
    st.subheader(f"Top {num_items_to_show} Complaint Types in {selected_district_breakdown}")
    st.dataframe(complaint_counts)

    # Optional: Display a bar chart of complaint type counts
    fig_type, ax_type = plt.subplots()
    ax_type.bar(complaint_counts['Complaint Type'], complaint_counts['Count'])
    ax_type.set_xlabel("Complaint Type")
    ax_type.set_ylabel("Number of Problems")
    ax_type.set_title(f"Top {num_items_to_show} Complaint Types in {selected_district_breakdown}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_type)

# New Section for Monthly Complaint Trend
st.header("Monthly Complaint Trend")

# Sidebar controls for the trend chart
st.sidebar.subheader("Controls for Complaint Trend")
trend_filter_type = st.sidebar.radio("Filter Trend by", ["All", "Organization", "Complaint Type", "District"], key="trend_filter_type")
selected_trend_option = None
if trend_filter_type == "Organization":
    selected_trend_option = st.sidebar.selectbox("Select Organization", ["All"] + sorted(organizations), key="trend_org")
elif trend_filter_type == "Complaint Type":
    selected_trend_option = st.sidebar.selectbox("Select Complaint Type", ["All"] + sorted(complaint_single_types), key="trend_type")
elif trend_filter_type == "District":
    selected_trend_option = st.sidebar.selectbox("Select District", ["All"] + sorted(districts), key="trend_district")

@st.cache_data
def get_monthly_trend_data(df):
    trend_data = df.groupby('month_year').size().reset_index(name='count')
    trend_data['month_year'] = trend_data['month_year'].astype(str) # Convert to string for Altair
    trend_data['month_year'] = pd.to_datetime(trend_data['month_year'])
    return trend_data

# Prepare data for the trend chart
trend_data_all = get_monthly_trend_data(df)
trend_data = trend_data_all.copy() # Start with all data

if trend_filter_type == "Organization" and selected_trend_option != "All":
    trend_data = df[df['organization'] == selected_trend_option].groupby('month_year').size().reset_index(name='count')
    trend_data['month_year'] = trend_data['month_year'].astype(str)
elif trend_filter_type == "Complaint Type" and selected_trend_option != "All":
    trend_data = df[df['type'] == selected_trend_option].groupby('month_year').size().reset_index(name='count')
    trend_data['month_year'] = trend_data['month_year'].astype(str)
elif trend_filter_type == "District" and selected_trend_option != "All":
    trend_data = df[df['district'] == selected_trend_option].groupby('month_year').size().reset_index(name='count')
    trend_data['month_year'] = trend_data['month_year'].astype(str)

# Create the Altair chart
# Base chart
base = alt.Chart(trend_data).encode(
    x=alt.X('month_year', title='Month and Year'),
    y=alt.Y('count', title='Number of Complaints'),
)

# Actual line chart
line = base.mark_line(color='steelblue').encode(
    tooltip=['month_year', 'count']
)

# Regression line
trend = base.transform_regression(
    'month_year', 'count', method='linear'
).mark_line(color='red', strokeDash=[5,5]).encode(
    tooltip=['month_year', 'count']
)

# Combine both
chart = (line + trend).properties(
    title=f"Monthly Complaint Trend ({trend_filter_type}: {selected_trend_option if selected_trend_option else 'All'})"
).interactive()

st.altair_chart(chart, use_container_width=True)