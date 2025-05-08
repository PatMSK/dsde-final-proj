import pandas as pd
import streamlit as st
import pydeck as pdk
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MultipleLocator


st.header("Organizations in each district")
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv", usecols=["ticket_id", "organization", "district", "coords"])
    df_stats = pd.read_csv("district_stats.csv")
    # Clean and prepare data
    df[['lon', 'lat']] = df['coords'].str.split(",", expand=True).astype(float)
    df = df.dropna(subset=['lat', 'lon'])
    return df, df_stats

# Load data
df,df_stats = load_data()

# districts
districts = df['district'].dropna().unique()
selected_district = st.selectbox("เลือกเขต", sorted(districts))

filtered_df = df[df['district'] == selected_district].copy()

COLOR_PALETTE = [
    [255, 99, 132, 160],
    [54, 162, 235, 160],
    [255, 206, 86, 160],
    [75, 192, 192, 160],
    [153, 102, 255, 160],
    [255, 159, 64, 160],
    [199, 199, 199, 160],
    [255, 99, 255, 160],
    [100, 255, 100, 160],
    [0, 0, 0, 160],
]


#filtered df
orgs = filtered_df['organization'].unique()
color_map = {
    org: COLOR_PALETTE[i % len(COLOR_PALETTE)]
    for i, org in enumerate(orgs)
}

filtered_df["color"] = filtered_df["organization"].map(color_map)

@st.cache_data
def each_district_scatter(filtered_df):
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


org_counts = filtered_df['organization'].value_counts().reset_index()
org_counts.columns = ['Organization', 'Count']

st.subheader("Organization counts")
st.dataframe(org_counts)

# scatter show each district
layer, view_state = each_district_scatter(filtered_df)
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "องค์กร: {organization}\nเขต: {district}"}
))


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

sort_order = st.radio("Select Rank", ["Top", "Bottom"])
num_districts = st.slider("District Count", min_value=5, max_value=20, value=10)

for metric in metrics:
    st.subheader(metric)
    plot_metric_bar_chart(df_stats, metric, sort_order, num_districts, max_metric_values[metric], min_metric_values[metric])
