import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

# ── 1.  Paths & demo data ──────────────────────────────────────────────
# BASE = os.getcwd()
# PERSONS_DIR = os.path.join(BASE, "Cleaned", "Persons", "Individuals")
BASE = "../data/"
PERSONS_DIR = "../data/preprocessed-data/individuals"

age      = pd.read_csv(os.path.join(PERSONS_DIR, "Age_Perfect_5yrs.csv"))
age_min, age_max = age["total"].min(), age["total"].max()

msoa_fp = os.path.join(BASE, "geodata", "MSOA_2021_EW_BGC_V3.shp")
red_fp  = os.path.join(BASE, "geodata", "boundary.geojson")        # red outline
blue_fp = os.path.join(BASE, "geodata", "boundary2.geojson")       # blue outline

# ── 2.  Read layers in WGS84 ───────────────────────────────────────────
gdf_msoa = gpd.read_file(msoa_fp).to_crs(4326)
red_bnd  = gpd.read_file(red_fp).to_crs(4326)
blue_bnd = gpd.read_file(blue_fp).to_crs(4326)

# ── 3.  Keep polygons that touch red boundary & remove two MSOAs ──────
gdf_msoa["total"] = np.random.randint(age_min, age_max, gdf_msoa.shape[0])
red_union = red_bnd.unary_union
gdf_clip  = gdf_msoa[gdf_msoa.intersects(red_union)].copy()

# drop areas that should stay blank
# exclude_codes = ["E02007116", "E02007021"]
# gdf_clip = gdf_clip[~gdf_clip["MSOA21CD"].isin(exclude_codes)].copy()

# ── 4.  Accurate centroids (project → centroid → back to WGS84) ───────
proj_crs = 27700
gdf_clip_proj = gdf_clip.to_crs(proj_crs)
centroids = (
    gdf_clip_proj.assign(geometry=gdf_clip_proj.geometry.centroid)
                .to_crs(4326)
)
centroids["lon"] = centroids.geometry.x
centroids["lat"] = centroids.geometry.y

# ── 5.  Choropleth layer ──────────────────────────────────────────────
fig = px.choropleth_mapbox(
    gdf_clip,
    geojson=json.loads(gdf_clip.to_json()),
    locations="MSOA21CD",
    featureidkey="properties.MSOA21CD",
    color="total",
    color_continuous_scale="tealrose",
    mapbox_style="carto-positron",
    center=dict(lat=51.7520, lon=-1.2577),
    zoom=10,
    opacity=0.7,
)

# ── 6.  Text trace for labels (only remaining MSOAs) ──────────────────
fig.add_trace(
    go.Scattermapbox(
        lon=centroids["lon"],
        lat=centroids["lat"],
        mode="text",
        text=centroids["MSOA21CD"],
        textfont=dict(size=10, color="black"),
        hoverinfo="text",
        showlegend=False,
    )
)

# ── 7.  Boundary overlays ─────────────────────────────────────────────
fig.update_layout(
    mapbox=dict(
        layers=[
            dict(source=json.loads(red_bnd.to_json()),  type="line",
                 color="red",  opacity=1, line=dict(width=3)),
            dict(source=json.loads(blue_bnd.to_json()), type="line",
                 color="blue", opacity=1, line=dict(width=3)),
        ]
    ),
    margin=dict(l=0, r=0, t=0, b=0)
)

fig.show()
