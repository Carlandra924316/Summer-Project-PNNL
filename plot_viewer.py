# conda activate thermocldphase_poster
# cd notebooks/fy24
# streamlit run plot_viewer.py --server.port=8508 --server.fileWatcherType poll

# TODO: Make this run on the aggregate.py output; parquet files. Hopefully much faster.
# IDEA: KDE plot for distribution of phase types


from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

BASE_PATH = Path("/data/home/levin/data/datastream/")

LIQUID = "liquid"
ICE = "ice"
MIXED = "mixed"
DRIZZLE = "drizzle"
LIQ_DRIZ = "liq_driz"
RAIN = "rain"
SNOW = "snow"
UNKNOWN = "unknown"

color_mapping = {
    LIQUID: "green",
    ICE: "blue",
    MIXED: "red",
    DRIZZLE: "lightseagreen",
    LIQ_DRIZ: "pink",
    RAIN: "gold",
    SNOW: "black",
    UNKNOWN: "dimgray",
}

st.set_page_config(page_title="Plot viewer", layout="wide")


phase_map = {
    0: "clear",
    1: "liquid",
    2: "ice",
    3: "mixed",
    4: "drizzle",
    5: "liq_driz",
    6: "rain",
    7: "snow",
    8: "unknown",
}

heatmap_colorscale = [
    [0 / 9, "white"],
    [1 / 9, "white"],
    [1 / 9, "green"],
    [2 / 9, "green"],
    [2 / 9, "blue"],
    [3 / 9, "blue"],
    [3 / 9, "red"],
    [4 / 9, "red"],
    [4 / 9, "turquoise"],
    [5 / 9, "turquoise"],
    [5 / 9, "orange"],
    [6 / 9, "orange"],
    [6 / 9, "gold"],
    [7 / 9, "gold"],
    [7 / 9, "black"],
    [8 / 9, "black"],
    [8 / 9, "gray"],
    [9 / 9, "gray"],
]


# fmt: off
heights = [0.16,0.19,0.22,0.25,0.28,0.31,0.34,0.37,0.4,0.43,0.46,0.49,0.52,0.55,0.58,0.61,0.64,0.67,0.7,0.73,0.76,0.79,0.82,0.85,0.88,0.91,0.94,0.97,1.0,1.03,1.06,1.09,1.12,1.15,1.18,1.21,1.24,1.27,1.3,1.33,1.36,1.39,1.42,1.45,1.48,1.51,1.54,1.57,1.6,1.63,1.66,1.69,1.72,1.75,1.78,1.81,1.84,1.87,1.9,1.93,1.96,1.99,2.02,2.05,2.08,2.11,2.14,2.17,2.2,2.23,2.26,2.29,2.32,2.35,2.38,2.41,2.44,2.47,2.5,2.53,2.56,2.59,2.62,2.65,2.68,2.71,2.74,2.77,2.8,2.83,2.86,2.89,2.92,2.95,2.98,3.01,3.04,3.07,3.1,3.13,3.16,3.19,3.22,3.25,3.28,3.31,3.34,3.37,3.4,3.43,3.46,3.49,3.52,3.55,3.58,3.61,3.64,3.67,3.7,3.73,3.76,3.79,3.82,3.85,3.88,3.91,3.94,3.97,4.0,4.03,4.06,4.09,4.12,4.15,4.18,4.21,4.24,4.27,4.3,4.33,4.36,4.39,4.42,4.45,4.48,4.51,4.54,4.57,4.6,4.63,4.66,4.69,4.72,4.75,4.78,4.81,4.84,4.87,4.9,4.93,4.96,4.99,5.02,5.05,5.08,5.11,5.14,5.17,5.2,5.23,5.26,5.29,5.32,5.35,5.38,5.41,5.44,5.47,5.5,5.53,5.56,5.59,5.62,5.65,5.68,5.71,5.74,5.77,5.8,5.83,5.86,5.89,5.92,5.95,5.98,6.01,6.04,6.07,6.1,6.13,6.16,6.19,6.22,6.25,6.28,6.31,6.34,6.37,6.4,6.43,6.46,6.49,6.52,6.55,6.58,6.61,6.64,6.67,6.7,6.73,6.76,6.79,6.82,6.85,6.88,6.91,6.94,6.97,7.0,7.03,7.06,7.09,7.12,7.15,7.18,7.21,7.24,7.27,7.3,7.33,7.36,7.39,7.42,7.45,7.48,7.51,7.54,7.57,7.6,7.63,7.66,7.69,7.72,7.75,7.78,7.81,7.84,7.87,7.9,7.93,7.96,7.99,8.02,8.05,8.08,8.11,8.14,8.17,8.2,8.23,8.26,8.29,8.32,8.35,8.38,8.41,8.44,8.47,8.5,8.53,8.56,8.59,8.62,8.65,8.68,8.71,8.74,8.77,8.8,8.83,8.86,8.89,8.92,8.95,8.98,9.01,9.04,9.07,9.1,9.13,9.16,9.19,9.22,9.25,9.28,9.31,9.34,9.37,9.4,9.43,9.46,9.49,9.52,9.55,9.58,9.61,9.64,9.67,9.7,9.73,9.76,9.79,9.82,9.85,9.88,9.91,9.94,9.97,10.0,10.03,10.06,10.09,10.12,10.15,10.18,10.21,10.24,10.27,10.3,10.33,10.36,10.39,10.42,10.45,10.48,10.51,10.54,10.57,10.6,10.63,10.66,10.69,10.72,10.75,10.78,10.81,10.84,10.87,10.9,10.93,10.96,10.99,11.02,11.05,11.08,11.11,11.14,11.17,11.2,11.23,11.26,11.29,11.32,11.35,11.38,11.41,11.44,11.47,11.5,11.53,11.56,11.59,11.62,11.65,11.68,11.71,11.74,11.77,11.8,11.83,11.86,11.89,11.92,11.95,11.98,12.01,12.04,12.07,12.1,12.13,12.16,12.19,12.22,12.25,12.28,12.31,12.34,12.37,12.4,12.43,12.46,12.49,12.52,12.55,12.58,12.61,12.64,12.67,12.7,12.73,12.76,12.79,12.82,12.85,12.88,12.91,12.94,12.97,13.0,13.03,13.06,13.09,13.12,13.15,13.18,13.21,13.24,13.27,13.3,13.33,13.36,13.39,13.42,13.45,13.48,13.51,13.54,13.57,13.6,13.63,13.66,13.69,13.72,13.75,13.78,13.81,13.84,13.87,13.9,13.93,13.96,13.99,14.02,14.05,14.08,14.11,14.14,14.17,14.2,14.23,14.26,14.29,14.32,14.35,14.38,14.41,14.44,14.47,14.5,14.53,14.56,14.59,14.62,14.65,14.68,14.71,14.74,14.77,14.8,14.83,14.86,14.89,14.92,14.95,14.98,15.01,15.04,15.07,15.1,15.13,15.16,15.19,15.22,15.25,15.28,15.31,15.34,15.37,15.4,15.43,15.46,15.49,15.52,15.55,15.58,15.61,15.64,15.67,15.7,15.73,15.76,15.79,15.82,15.85,15.88,15.91,15.94,15.97,16.0,16.03,16.06,16.09,16.12,16.15,16.18,16.21,16.24,16.27,16.3,16.33,16.36,16.39,16.42,16.45,16.48,16.51,16.54,16.57,16.6,16.63,16.66,16.69,16.72,16.75,16.78,16.81,16.84,16.87,16.9,16.93,16.96,16.99,17.02,17.05,17.08,17.11,17.14,17.17,17.2,17.23,17.26,17.29,17.32,17.35,17.38,17.41,17.44,17.47,17.5,17.53,17.56,17.59,17.62,17.65,17.68,17.71,17.74,17.77,17.8,17.83,17.86,17.89,17.92]  # type: ignore
# fmt: on


def plot_cloud_phase(ds: xr.Dataset) -> go.Figure:
    fig = px.imshow(
        ds["cloud_phase_mplgr"].T.fillna(0).astype(int),
        origin="lower",
        labels=dict(x="time", y="height", color="phase"),
        color_continuous_scale=heatmap_colorscale,
        range_color=(0, 8),
        title=file.name,
    )
    fig = fig.update_coloraxes(
        colorbar_tickmode="array",
        colorbar_tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        colorbar_ticktext=[
            "clear",
            "liquid",
            "ice",
            "mixed",
            "drizzle",
            "liquid-drizzle",
            "rain",
            "snow",
            "unknown",
        ],
        colorbar_outlinewidth=1,
        colorbar_outlinecolor="black",
    )
    return fig


@st.cache_data
def get_files_and_dates(folder: Path) -> tuple[list[Path], list[datetime]]:
    files = sorted(folder.glob("*.nc"))
    dates = [datetime.strptime(file.name.split(".")[2], "%Y%m%d") for file in files]
    return files, dates


# ######################################################################################
# ######################################################################################
# ######################################################################################


# @st.cache_data
# def load_pq_data() -> pd.DataFrame:
#     pq_file = Path("fy24_data/agg/phase/cor/cor.20190201.000000.parquet")
#     data = pd.read_parquet(pq_file)
#     data = data.reset_index()
#     data["time"] = pd.to_datetime(data["time"])
#     # data["height"] = data["height"].round(2)
#     data["height"] = data["height"].apply(lambda x: round(x, 2))
#     data = data.set_index(["time", "height"])
#     data["text_labels"] = (
#         data["cloud_phase_mplgr"].astype(int).map(phase_map).astype("category")
#     )
#     return data


# pq = load_pq_data()
# df = pq.loc[(slice("2019-02-01", "2019-02-02"), slice(None)), :]

# # Create a full grid
# time_range = pd.date_range(
#     df.index.get_level_values("time").min(),
#     df.index.get_level_values("time").max(),
#     freq="30min",
# )
# height_range = heights
# fig = go.Figure(
#     data=go.Heatmap(
#         z=df["cloud_phase_mplgr"],
#         x=df.index.get_level_values("time"),
#         y=df.index.get_level_values("height"),
#         text=df["text_labels"],
#         hoverinfo="text",
#         colorscale=heatmap_colorscale,
#         colorbar=dict(
#             tickvals=list(phase_map.keys()), ticktext=list(phase_map.values())
#         ),
#     )
# )
# fig.update_layout(
#     xaxis_title="Time",
#     yaxis_title="Height (km)",
# )
# st.plotly_chart(fig, use_container_width=True)
# st.stop()

# ######################################################################################
# ######################################################################################
# ######################################################################################

with st.sidebar.container():
    site_folder_map = {
        folder.parent.name: folder
        for folder in sorted(BASE_PATH.glob("*/*thermocldphase*"))
    }

    site = st.selectbox("Site", site_folder_map)
    files, dates = get_files_and_dates(site_folder_map[site])

    _ = st.select_slider(
        "Date",
        range(len(dates)),
        format_func=lambda i: dates[i].strftime("%Y-%m-%d"),
        key="index",
    )
    low, high = st.slider("Height Range", 0.0, 18.0, (0.0, 5.0), 0.1)


def update_index(delta: int):
    st.session_state["index"] += delta


button_cols = st.sidebar.columns(3)
button_cols[0].button(
    "Previous",
    on_click=lambda: update_index(-1),
    disabled=st.session_state["index"] == 0,
)
button_cols[-1].button(
    "Next",
    on_click=lambda: update_index(1),
    disabled=st.session_state["index"] == len(files) - 1,
)

# LOAD THE DATA
index = st.session_state["index"]
file = files[index]
date = dates[index]
ds = xr.open_dataset(file)
ds = ds.where((ds.height >= low) & (ds.height <= high), drop=True)


var = st.sidebar.selectbox(
    "Extra var to plot",
    [""] + [v for v in sorted(ds.data_vars) if ds[v].dims == ("time", "height")],
)
log = st.sidebar.checkbox("log")
if var:  # make var plot and phase plot side-by-side
    chart_cols = st.columns((1, 1))
    with chart_cols[0]:
        fig = plot_cloud_phase(ds)
        st.plotly_chart(fig, use_container_width=True)
    with chart_cols[1]:
        plot = st.empty()

        if var:
            fig = px.imshow(
                np.log10(ds[var].T) if log else ds[var].T,
                origin="lower",
                title=var,
            )
            plot.plotly_chart(fig, use_container_width=True)
else:  # make the phase chart big
    fig = plot_cloud_phase(ds)
    st.plotly_chart(fig, use_container_width=True)


df = ds["cloud_phase_mplgr"].to_dataframe()
df = df.where(lambda row: row.cloud_phase_mplgr != 0, axis=0).dropna()
st.plotly_chart(px.histogram(df), use_container_width=True)
