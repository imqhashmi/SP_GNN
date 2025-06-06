import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm
from torch.nn import CrossEntropyLoss
import random
import time
from datetime import timedelta
import json
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import argparse
import geopandas as gpd  # Added for geo plotting
import plotly.express as px  # Added for geo plotting

# Add argument parser for command line parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate synthetic individuals using GNN')
    parser.add_argument('--area_code', type=str, required=True,
                       help='Oxford area code to process (e.g., E02005924)')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
selected_area_code = args.area_code

print(f"Running Individual Generation for area: {selected_area_code}")

# Device selection with better fallback options
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                      'cpu')
print(f"Using device: {device}")

def create_geo_plot_trace(selected_area_code, current_dir):
    """
    Create a geo plot trace showing all areas in white except the selected area code which is shaded.
    Returns the geo traces that can be added to a subplot.
    
    Parameters:
    selected_area_code - The area code to highlight
    current_dir - The current directory path for file loading
    """
    try:
        # Define paths relative to the code directory
        BASE = os.path.join(current_dir, '../data/')
        PERSONS_DIR = os.path.join(current_dir, '../data/preprocessed-data/individuals')
        
        # Load age data and rename the geography code
        age_file = os.path.join(PERSONS_DIR, "Age_Perfect_5yrs.csv")
        if not os.path.exists(age_file):
            print(f"Warning: Age data file not found at {age_file}")
            return [], {}
            
        age = pd.read_csv(age_file)
        age = age.rename(columns={"geography code": "MSOA21CD"})
        
        # Load shapefiles
        msoa_fp = os.path.join(BASE, "geodata", "MSOA_2021_EW_BGC_V3.shp")
        red_fp = os.path.join(BASE, "geodata", "boundary.geojson")
        
        if not os.path.exists(msoa_fp) or not os.path.exists(red_fp):
            print(f"Warning: Geodata files not found. Expected at {msoa_fp} and {red_fp}")
            return [], {}
        
        # Read in spatial data
        gdf_msoa = gpd.read_file(msoa_fp).to_crs(4326)
        red_bnd = gpd.read_file(red_fp).to_crs(4326)
        
        # Merge population totals
        gdf_msoa = gdf_msoa.merge(age[["MSOA21CD", "total"]], on="MSOA21CD", how="left")
        gdf_msoa["total"] = gdf_msoa["total"].fillna(0)
        
        # Manually remove unwanted MSOAs
        exclude_codes = [
            "E02005939", "E02005979", "E02005963", "E02005959"
        ]
        gdf_msoa = gdf_msoa[~gdf_msoa["MSOA21CD"].isin(exclude_codes)]
        
        # Clip to red boundary
        red_union = red_bnd.unary_union
        gdf_clip = gdf_msoa[gdf_msoa.intersects(red_union)].copy()
        
        # Create color column: selected area gets color 1, others get color 0
        gdf_clip["color_value"] = gdf_clip["MSOA21CD"].apply(
            lambda x: 1 if x == selected_area_code else 0
        )
        
        # Compute accurate centroids for labels
        proj_crs = 27700
        gdf_proj = gdf_clip.to_crs(proj_crs)
        centroids = gdf_proj.geometry.centroid.to_crs(4326)
        gdf_clip["lon"] = centroids.x
        gdf_clip["lat"] = centroids.y
        
        # Create traces list
        traces = []
        
        # Add choropleth trace
        choropleth_trace = go.Choropleth(
            geojson=json.loads(gdf_clip.to_json()),
            locations=gdf_clip["MSOA21CD"],
            featureidkey="properties.MSOA21CD",
            z=gdf_clip["color_value"],
            colorscale=[[0, "white"], [1, "lightblue"]],  # White for 0, light blue for selected
            showscale=False,
            hovertemplate="<b>%{location}</b><extra></extra>",
            name="Areas"
        )
        traces.append(choropleth_trace)
        
        # Add red boundary outline
        for poly in red_bnd.geometry.explode(index_parts=False):
            boundary_trace = go.Scattergeo(
                lon=list(poly.exterior.coords.xy[0]),
                lat=list(poly.exterior.coords.xy[1]),
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
                hoverinfo="skip"
            )
            traces.append(boundary_trace)
        
        # Calculate bounds for the geo layout with reduced padding for bigger geo plots
        # Use red boundary bounds instead of just clipped areas for better coverage
        red_bounds = red_bnd.total_bounds  # [minx, miny, maxx, maxy]
        lon_min, lat_min, lon_max, lat_max = red_bounds
        
        # Reduce padding to make geo plots bigger within their allocated space
        lat_padding = (lat_max - lat_min) * 0.1  # Reduced to 10% padding (was 30%)
        lon_padding = (lon_max - lon_min) * 0.1  # Reduced to 10% padding (was 30%)
        
        geo_layout = {
            'visible': False,
            'lataxis_range': [lat_min - lat_padding, lat_max + lat_padding],
            'lonaxis_range': [lon_min - lon_padding, lon_max + lon_padding],
            'projection_type': 'mercator'
        }
        
        return traces, geo_layout
        
    except Exception as e:
        print(f"Warning: Could not create geo plot trace: {e}")
        return [], {}

def get_target_tensors(cross_table, feature_1_categories, feature_1_map, feature_2_categories, feature_2_map, feature_3_categories, feature_3_map):
    y_feature_1 = torch.zeros(num_persons, dtype=torch.long, device=device)
    y_feature_2 = torch.zeros(num_persons, dtype=torch.long, device=device)
    y_feature_3 = torch.zeros(num_persons, dtype=torch.long, device=device)
    
    # Populate target tensors based on the cross table and feature categories
    person_idx = 0
    for _, row in cross_table.iterrows():
        for feature_1 in feature_1_categories:
            for feature_2 in feature_2_categories:
                for feature_3 in feature_3_categories:
                    col_name = f'{feature_1} {feature_2} {feature_3}'
                    count = int(row.get(col_name, 0))
                    for _ in range(count):
                        if person_idx < num_persons:
                            y_feature_1[person_idx] = feature_1_map.get(feature_1, -1)
                            y_feature_2[person_idx] = feature_2_map.get(feature_2, -1)
                            y_feature_3[person_idx] = feature_3_map.get(feature_3, -1)
                            person_idx += 1

    return (y_feature_1, y_feature_2, y_feature_3)


# Load the data from individual tables
current_dir = os.path.dirname(os.path.abspath(__file__))
age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Age_Perfect_5yrs.csv'))
sex_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Sex.csv'))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Ethnicity.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Religion.csv'))
marital_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Marital.csv'))
ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/EthnicityBySexByAge.csv'))
religion_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/ReligionbySexbyAge.csv'))
marital_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/MaritalbySexbyAgeModified.csv'))

# Define the Oxford areas
oxford_areas = [selected_area_code]
print(f"Processing Oxford area: {oxford_areas[0]}")

# Filter the DataFrame for the specified Oxford areas
age_df = age_df[age_df['geography code'].isin(oxford_areas)]
sex_df = sex_df[sex_df['geography code'].isin(oxford_areas)]
ethnicity_df = ethnicity_df[ethnicity_df['geography code'].isin(oxford_areas)]
religion_df = religion_df[religion_df['geography code'].isin(oxford_areas)]
marital_df = marital_df[marital_df['geography code'].isin(oxford_areas)]
ethnic_by_sex_by_age_df = ethnic_by_sex_by_age_df[ethnic_by_sex_by_age_df['geography code'].isin(oxford_areas)]
religion_by_sex_by_age_df = religion_by_sex_by_age_df[religion_by_sex_by_age_df['geography code'].isin(oxford_areas)]
marital_by_sex_by_age_df = marital_by_sex_by_age_df[marital_by_sex_by_age_df['geography code'].isin(oxford_areas)]

# Define the age groups, sex categories, and ethnicity categories
age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
sex_categories = ['M', 'F']
ethnicity_categories = ['W1', 'W2', 'W3', 'W4', 'M1', 'M2', 'M3', 'M4', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'O1', 'O2']
religion_categories = ['C','B','H','J','M','S','O','N','NS']
marital_categories = ['Single','Married','Partner','Separated','Divorced','Widowed']

# Encode the categories to indices
age_map = {category: i for i, category in enumerate(age_groups)}
sex_map = {category: i for i, category in enumerate(sex_categories)}
ethnicity_map = {category: i for i, category in enumerate(ethnicity_categories)}
religion_map = {category: i for i, category in enumerate(religion_categories)}
marital_map = {category: i for i, category in enumerate(marital_categories)}

# Total number of persons from the total column
num_persons = int(age_df['total'].sum())

print(f"Total number of persons: {num_persons}")

# Create person nodes with unique IDs
person_nodes = torch.arange(num_persons).view(num_persons, 1).to(device)

# Create nodes for age categories
age_nodes = torch.tensor([[age_map[age]] for age in age_groups], dtype=torch.float).to(device)

# Create nodes for sex categories
sex_nodes = torch.tensor([[sex_map[sex]] for sex in sex_categories], dtype=torch.float).to(device)

# Create nodes for ethnicity categories
ethnicity_nodes = torch.tensor([[ethnicity_map[ethnicity]] for ethnicity in ethnicity_categories], dtype=torch.float).to(device)

# Create nodes for religion categories
religion_nodes = torch.tensor([[religion_map[religion]] for religion in religion_categories], dtype=torch.float).to(device)

# Create nodes for marital categories
marital_nodes = torch.tensor([[marital_map[marital]] for marital in marital_categories], dtype=torch.float).to(device)

# Combine all nodes into a single tensor
node_features = torch.cat([person_nodes, age_nodes, sex_nodes, ethnicity_nodes, religion_nodes, marital_nodes], dim=0).to(device)

# Calculate the distribution for age categories
age_probabilities = age_df.drop(columns = ["geography code", "total"]) / num_persons
sex_probabilities = sex_df.drop(columns = ["geography code", "total"]) / num_persons
ethnicity_probabilities = ethnicity_df.drop(columns = ["geography code", "total"]) / num_persons
religion_probabilities = religion_df.drop(columns = ["geography code", "total"]) / num_persons
marital_probabilities = marital_df.drop(columns = ["geography code", "total"]) / num_persons

# New function to generate edge index
def generate_edge_index(num_persons):
    edge_index = []
    age_start_idx = num_persons
    sex_start_idx = age_start_idx + len(age_groups)
    ethnicity_start_idx = sex_start_idx + len(sex_categories)
    religion_start_idx = ethnicity_start_idx + len(ethnicity_categories)
    marital_start_idx = religion_start_idx + len(religion_categories)

    # Convert the probability series to a list of probabilities for sampling
    age_prob_list = age_probabilities.values.tolist()[0]
    sex_prob_list = sex_probabilities.values.tolist()[0]
    ethnicity_prob_list = ethnicity_probabilities.values.tolist()[0]
    religion_prob_list = religion_probabilities.values.tolist()[0]
    marital_prob_list = marital_probabilities.values.tolist()[0]

    for i in range(num_persons):
        # Sample the categories using weighted random sampling
        age_category = random.choices(range(age_start_idx, sex_start_idx), weights=age_prob_list, k=1)[0]
        sex_category = random.choices(range(sex_start_idx, ethnicity_start_idx), weights=sex_prob_list, k=1)[0]
        ethnicity_category = random.choices(range(ethnicity_start_idx, religion_start_idx), weights=ethnicity_prob_list, k=1)[0]
        religion_category = random.choices(range(religion_start_idx, marital_start_idx), weights=religion_prob_list, k=1)[0]
        marital_category = random.choices(range(marital_start_idx, marital_start_idx + len(marital_categories)), weights=marital_prob_list, k=1)[0]
        
        # Append edges for each category
        edge_index.append([i, age_category])
        edge_index.append([i, sex_category])
        edge_index.append([i, ethnicity_category])
        edge_index.append([i, religion_category])
        edge_index.append([i, marital_category])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    return edge_index

# Generate edge index using the new function
edge_index = generate_edge_index(num_persons)

# Create the data object for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index).to(device)

# Get target tensors
targets = []
targets.append(
    (
        ('sex', 'age', 'ethnicity'), 
        get_target_tensors(ethnic_by_sex_by_age_df, sex_categories, sex_map, age_groups, age_map, ethnicity_categories, ethnicity_map)
    )
)
targets.append(
    (
        ('sex', 'age', 'marital'), 
        get_target_tensors(marital_by_sex_by_age_df, sex_categories, sex_map, age_groups, age_map, marital_categories, marital_map)
    )
)
targets.append(
    (
        ('sex', 'age', 'religion'), 
        get_target_tensors(religion_by_sex_by_age_df, sex_categories, sex_map, age_groups, age_map, religion_categories, religion_map)
    )
)

class EnhancedGNNModelWithMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, mlp_hidden_dim, out_channels_age, out_channels_sex, out_channels_ethnicity, out_channels_religion, out_channels_marital, dropout_rate=0.5):
        super(EnhancedGNNModelWithMLP, self).__init__()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        
        # Batch normalization
        self.batch_norm1 = GraphNorm(hidden_channels)
        self.batch_norm2 = GraphNorm(hidden_channels)
        self.batch_norm3 = GraphNorm(hidden_channels)
        self.batch_norm4 = GraphNorm(hidden_channels)
        
        # Dropout layer
        # self.dropout = torch.nn.Dropout(0.1)
        
        # MLP for each output attribute
        self.mlp_age = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_age)
        )
        
        self.mlp_sex = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_sex)
        )
        
        self.mlp_ethnicity = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_ethnicity)
        )
        
        self.mlp_religion = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_religion)
        )
        
        self.mlp_marital = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_marital)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Pass through GraphSAGE layers
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv4(x, edge_index)
        x = self.batch_norm4(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        # Pass the node embeddings through the MLPs for final attribute predictions
        age_out = self.mlp_age(x)
        sex_out = self.mlp_sex(x)
        ethnicity_out = self.mlp_ethnicity(x)
        religion_out = self.mlp_religion(x)
        marital_out = self.mlp_marital(x)
        
        return age_out, sex_out, ethnicity_out, religion_out, marital_out

# Custom loss function
def custom_loss_function(first_out, second_out, third_out, y_first, y_second, y_third):
    first_pred = first_out.argmax(dim=1)
    second_pred = second_out.argmax(dim=1)
    third_pred = third_out.argmax(dim=1)
    loss_first = F.cross_entropy(first_out, y_first)
    loss_second = F.cross_entropy(second_out, y_second)
    loss_third = F.cross_entropy(third_out, y_third)
    total_loss = loss_first + loss_second + loss_third
    return total_loss

# Distribution-based accuracy functions
def create_predicted_crosstable(pred_1, pred_2, pred_3, categories_1, categories_2, categories_3):
    """
    Create a cross-table from predictions.
    
    Args:
        pred_1, pred_2, pred_3: Predicted class indices for the three attributes
        categories_1, categories_2, categories_3: Category lists for the three attributes
    
    Returns:
        Dictionary representing the predicted cross-table
    """
    # Create index combinations (feature_1 feature_2 feature_3)
    combinations = []
    for cat1 in categories_1:
        for cat2 in categories_2:
            for cat3 in categories_3:
                combinations.append(f'{cat1} {cat2} {cat3}')
    
    # Count occurrences of each combination in predictions
    predicted_counts = {}
    pred_1_names = [categories_1[i] for i in pred_1.cpu().numpy()]
    pred_2_names = [categories_2[i] for i in pred_2.cpu().numpy()]
    pred_3_names = [categories_3[i] for i in pred_3.cpu().numpy()]
    
    for combo in combinations:
        predicted_counts[combo] = 0
    
    for p1, p2, p3 in zip(pred_1_names, pred_2_names, pred_3_names):
        combo = f'{p1} {p2} {p3}'
        if combo in predicted_counts:
            predicted_counts[combo] += 1
    
    return predicted_counts

def calculate_r2_accuracy(generated_counts, target_counts):
    """
    Simple R² measure comparing two distributions:
    R² = 1 - (SSE / SST), with SSE = sum of squared errors
    """
    gen_vals = np.array(list(generated_counts.values()), dtype=float)
    tgt_vals = np.array(list(target_counts.values()), dtype=float)

    sse = np.sum((gen_vals - tgt_vals) ** 2)
    sst = np.sum((tgt_vals - tgt_vals.mean()) ** 2)

    return 1.0 - sse / sst if sst > 1e-12 else 1.0

# Define the hyperparameters to tune
# learning_rates = [0.001]
# hidden_channel_options = [64]
learning_rates = [0.001, 0.0005, 0.0001]
hidden_channel_options = [64, 128, 256]
mlp_hidden_dim = 128
num_epochs = 2500
# num_epochs = 10

# Results storage
results = []
time_results = []
best_model_info = {
    'model_state': None,
    'loss': float('inf'),
    'accuracy': 0,
    'predictions': None,
    'lr': None,
    'hidden_channels': None,
    'training_time': None
}

# Optimized GPU-friendly accuracy function for multi-task learning
def calculate_distribution_task_accuracy(pred_1, pred_2, pred_3, target_combination, actual_crosstable):
    """
    Fast GPU-optimized distribution-based accuracy calculation.
    Uses tensor operations instead of pandas for speed during training.
    """
    categories_1, categories_2, categories_3 = target_combination
    
    # Map attribute names to category counts
    category_sizes = {
        'sex': len(sex_categories),
        'age': len(age_groups), 
        'ethnicity': len(ethnicity_categories),
        'religion': len(religion_categories),
        'marital': len(marital_categories)
    }
    
    size_1 = category_sizes[categories_1]
    size_2 = category_sizes[categories_2] 
    size_3 = category_sizes[categories_3]
    
    # Create predicted counts tensor (keep on GPU)
    # Use a flattened approach: combination_idx = pred_1 * (size_2 * size_3) + pred_2 * size_3 + pred_3
    combo_indices = pred_1 * (size_2 * size_3) + pred_2 * size_3 + pred_3
    total_combinations = size_1 * size_2 * size_3
    
    # Count occurrences efficiently on GPU
    predicted_counts = torch.bincount(combo_indices, minlength=total_combinations).float()
    
    # Pre-compute actual counts tensor (do this only once, not every epoch)
    if not hasattr(calculate_distribution_task_accuracy, f'actual_counts_{categories_3}'):
        # Extract actual counts and convert to tensor format
        actual_counts_tensor = torch.zeros(total_combinations, dtype=torch.float, device=device)
        
        category_map = {
            'sex': sex_categories,
            'age': age_groups, 
            'ethnicity': ethnicity_categories,
            'religion': religion_categories,
            'marital': marital_categories
        }
        
        cats_1 = category_map[categories_1]
        cats_2 = category_map[categories_2] 
        cats_3 = category_map[categories_3]
        
        for i1, cat1 in enumerate(cats_1):
            for i2, cat2 in enumerate(cats_2):
                for i3, cat3 in enumerate(cats_3):
                    original_col = f'{cat1} {cat2} {cat3}'
                    if original_col in actual_crosstable.columns:
                        combo_idx = i1 * (size_2 * size_3) + i2 * size_3 + i3
                        actual_counts_tensor[combo_idx] = actual_crosstable[original_col].iloc[0]
        
        # Cache the result to avoid recomputation
        setattr(calculate_distribution_task_accuracy, f'actual_counts_{categories_3}', actual_counts_tensor)
    
    actual_counts = getattr(calculate_distribution_task_accuracy, f'actual_counts_{categories_3}')
    
    # Calculate R² efficiently on GPU
    actual_mean = actual_counts.mean()
    ss_tot = torch.sum((actual_counts - actual_mean) ** 2)
    ss_res = torch.sum((actual_counts - predicted_counts) ** 2)
    
    if ss_tot > 1e-12:
        r2 = 1.0 - (ss_res / ss_tot)
        return max(0.0, r2.item())  # Ensure non-negative and convert to Python float
    else:
        return 1.0

# Define a function to train the model
def train_model(lr, hidden_channels, num_epochs, data, targets):
    # Initialize model, optimizer, and loss functions
    model = EnhancedGNNModelWithMLP(
        in_channels=node_features.size(1),
        hidden_channels=hidden_channels,
        mlp_hidden_dim=mlp_hidden_dim,
        out_channels_age=len(age_groups),
        out_channels_sex=len(sex_categories),
        out_channels_ethnicity=len(ethnicity_categories),
        out_channels_religion=len(religion_categories),
        out_channels_marital=len(marital_categories)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Track best epoch state
    best_epoch_loss = float('inf')
    best_epoch_state = None
    loss_data = {}
    accuracy_data = {}
    
    # Storage for tracking metrics across epochs
    epoch_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        age_out, sex_out, ethnicity_out, religion_out, marital_out = model(data)

        out = {}
        out['age'] = age_out[:num_persons]  # Only take person nodes' outputs
        out['sex'] = sex_out[:num_persons]
        out['ethnicity'] = ethnicity_out[:num_persons]
        out['religion'] = religion_out[:num_persons]
        out['marital'] = marital_out[:num_persons]

        loss = 0
        
        # Calculate losses for all target combinations
        for i in range(len(targets)):
            current_loss = custom_loss_function(
                out[targets[i][0][0]], out[targets[i][0][1]], out[targets[i][0][2]],
                targets[i][1][0], targets[i][1][1], targets[i][1][2]
            )
            loss += current_loss
            
        # Calculate accuracy only every 100 epochs to speed up training
        if (epoch + 1) % 100 == 0:
            epoch_task_accuracies = []
            for i in range(len(targets)):
                # Calculate distribution-based accuracy for this task
                pred_1 = out[targets[i][0][0]].argmax(dim=1)
                pred_2 = out[targets[i][0][1]].argmax(dim=1)
                pred_3 = out[targets[i][0][2]].argmax(dim=1)
                
                # Get the corresponding actual cross-table
                if i == 0:  # sex-age-ethnicity
                    actual_crosstable = ethnic_by_sex_by_age_df
                elif i == 1:  # sex-age-marital  
                    actual_crosstable = marital_by_sex_by_age_df
                else:  # sex-age-religion
                    actual_crosstable = religion_by_sex_by_age_df
                
                task_distribution_accuracy = calculate_distribution_task_accuracy(
                    pred_1, pred_2, pred_3, targets[i][0], actual_crosstable
                )
                epoch_task_accuracies.append(task_distribution_accuracy)

            # Calculate average accuracy for this epoch
            avg_epoch_accuracy = sum(epoch_task_accuracies) / len(epoch_task_accuracies)
            epoch_accuracies.append(avg_epoch_accuracy)
            
            # Print metrics every 100 epochs
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Distribution Accuracy: {avg_epoch_accuracy:.4f}')
            
        # Store best epoch state
        if loss.item() < best_epoch_loss:
            best_epoch_loss = loss.item()
            best_epoch_state = model.state_dict().copy()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Store loss data for each epoch
        loss_data[epoch] = loss.item()

    # Calculate average accuracy across all epochs
    average_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
            
    # Load best epoch state for evaluation
    model.load_state_dict(best_epoch_state)

    # Evaluate accuracy after training
    model.eval()
    with torch.no_grad():
        age_out, sex_out, ethnicity_out, religion_out, marital_out = model(data)
        
        out = {}
        out['age'] = age_out[:num_persons]
        out['sex'] = sex_out[:num_persons]
        out['ethnicity'] = ethnicity_out[:num_persons]
        out['religion'] = religion_out[:num_persons]
        out['marital'] = marital_out[:num_persons]
        
        age_pred = out['age'].argmax(dim=1)
        sex_pred = out['sex'].argmax(dim=1)
        ethnicity_pred = out['ethnicity'].argmax(dim=1)
        religion_pred = out['religion'].argmax(dim=1)
        marital_pred = out['marital'].argmax(dim=1)

        # Calculate distribution-based accuracy across all tasks
        net_accuracy = 0
        final_task_accuracies = {}
        
        for i in range(len(targets)):
            pred_1 = out[targets[i][0][0]].argmax(dim=1)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)
            pred_3 = out[targets[i][0][2]].argmax(dim=1)
            
            # Get the corresponding actual cross-table
            if i == 0:  # sex-age-ethnicity
                actual_crosstable = ethnic_by_sex_by_age_df
            elif i == 1:  # sex-age-marital  
                actual_crosstable = marital_by_sex_by_age_df
            else:  # sex-age-religion
                actual_crosstable = religion_by_sex_by_age_df
            
            # Calculate distribution-based accuracy (R²)
            task_distribution_accuracy = calculate_distribution_task_accuracy(
                pred_1, pred_2, pred_3, targets[i][0], actual_crosstable
            )
            
            net_accuracy += task_distribution_accuracy
            task_name = '_'.join(targets[i][0])
            final_task_accuracies[task_name] = task_distribution_accuracy * 100
        
        final_accuracy = net_accuracy / len(targets)
        
        # Print final task accuracies
        print(f"\n=== DISTRIBUTION-BASED ACCURACY RESULTS ===")
        for task, acc in final_task_accuracies.items():
            print(f"{task} distribution accuracy (R²): {acc:.2f}%")
        print(f"Overall distribution accuracy: {final_accuracy*100:.2f}%")
        
        # Update best model info if this model performs better
        global best_model_info
        if final_accuracy > best_model_info['accuracy'] or (final_accuracy == best_model_info['accuracy'] and best_epoch_loss < best_model_info['loss']):
            best_model_info.update({
                'model_state': best_epoch_state,
                'loss': best_epoch_loss,
                'accuracy': final_accuracy,
                'predictions': (sex_pred, age_pred, ethnicity_pred, religion_pred, marital_pred),
                'lr': lr,
                'hidden_channels': hidden_channels
            })

        # Return the final loss, average accuracy across epochs, and final accuracies
        return best_epoch_loss, average_accuracy, final_accuracy, (sex_pred, age_pred, ethnicity_pred, religion_pred, marital_pred)

# Run the grid search over hyperparameters
total_start_time = time.time()
time_results = []

for lr in learning_rates:
    for hidden_channels in hidden_channel_options:
        print(f"Training with lr={lr}, hidden_channels={hidden_channels}")
        
        # Start timing for this combination
        start_time = time.time()
        
        # Train the model for the current combination of hyperparameters
        final_loss, average_accuracy, final_accuracy, predictions = train_model(lr, hidden_channels, num_epochs, data, targets)
        
        # End timing for this combination
        end_time = time.time()
        train_time = end_time - start_time
        train_time_str = str(timedelta(seconds=int(train_time)))
        
        # Store the results
        results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'final_loss': final_loss,
            # 'average_distribution_accuracy': average_accuracy,
            'average_accuracy': final_accuracy,
            'training_time': train_time_str
        })
        
        # Store timing results
        time_results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'training_time': train_time_str
        })

        # Print the results for the current run
        print(f"Finished training with lr={lr}, hidden_channels={hidden_channels}")
        print(f"Final Loss: {final_loss}, Average Distribution Accuracy: {average_accuracy:.4f}, Final Distribution Accuracy: {final_accuracy:.4f}")
        print(f"Training time: {train_time_str}")

# Calculate total training time
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
total_training_time_str = str(timedelta(seconds=int(total_training_time)))
print(f"Total training time: {total_training_time_str}")

# After all runs, display results
results_df = pd.DataFrame(results)
print("\nHyperparameter tuning results:")
print(results_df)

# Print best model information
print("\nBest Model Information:")
print(f"Learning Rate: {best_model_info['lr']}")
print(f"Hidden Channels: {best_model_info['hidden_channels']}")
print(f"Best Loss: {best_model_info['loss']:.4f}")
print(f"Best Distribution Accuracy (R²): {best_model_info['accuracy']:.4f}")

# Create output directory if it doesn't exist
output_dir = os.path.join(current_dir, 'outputs', f'individuals_{selected_area_code}')
os.makedirs(output_dir, exist_ok=True)

# Save best model state
# torch.save(best_model_info['model_state'], os.path.join(output_dir, 'best_individual_model_state.pt'))

# Save best model predictions
best_predictions = {
    'sex_pred': best_model_info['predictions'][0].cpu().numpy(),
    'age_pred': best_model_info['predictions'][1].cpu().numpy(),
    'ethnicity_pred': best_model_info['predictions'][2].cpu().numpy(),
    'religion_pred': best_model_info['predictions'][3].cpu().numpy(),
    'marital_pred': best_model_info['predictions'][4].cpu().numpy()
}
# np.save(os.path.join(output_dir, 'best_individual_model_predictions.npy'), best_predictions)

# Save hyperparameter results
results_df.to_csv(os.path.join(output_dir, 'generateIndividuals_results.csv'), index=False)

# Save best model configuration
best_config = {
    'learning_rate': best_model_info['lr'],
    'hidden_channels': best_model_info['hidden_channels'],
    'loss': best_model_info['loss'],
    'accuracy': best_model_info['accuracy']
}
# with open(os.path.join(output_dir, 'best_individual_model_config.json'), 'w') as f:
#     json.dump(best_config, f, indent=4)

# Extract the best model's predictions for visualization
sex_pred, age_pred, ethnicity_pred, religion_pred, marital_pred = best_model_info['predictions']

# Create person tensor with attributes
# Format: [person_id, age, religion, ethnicity, marital, sex]
person_tensor = torch.zeros(num_persons, 6, device=device)
person_tensor[:, 0] = torch.arange(num_persons, device=device)  # Person ID
person_tensor[:, 1] = age_pred  # Age
person_tensor[:, 2] = religion_pred  # Religion
person_tensor[:, 3] = ethnicity_pred  # Ethnicity
person_tensor[:, 4] = marital_pred  # Marital status
person_tensor[:, 5] = sex_pred  # Sex

# Save person tensor
person_tensor_path = os.path.join(output_dir, 'person_nodes.pt')
torch.save(person_tensor.cpu(), person_tensor_path)
print(f"\nBest model outputs saved to {output_dir}")

sex_pred_names = [sex_categories[i] for i in sex_pred.cpu().numpy()]
age_pred_names = [age_groups[i] for i in age_pred.cpu().numpy()]
ethnicity_pred_names = [ethnicity_categories[i] for i in ethnicity_pred.cpu().numpy()]
religion_pred_names = [religion_categories[i] for i in religion_pred.cpu().numpy()]
marital_pred_names = [marital_categories[i] for i in marital_pred.cpu().numpy()]

# Calculate actual distributions
sex_actual = {}
age_actual = {}
ethnicity_actual = {}
religion_actual = {}
marital_actual = {}

# Extract counts from the original data frames
for sex in sex_categories:
    sex_actual[sex] = sex_df[sex].iloc[0]

for age in age_groups:
    age_actual[age] = age_df[age].iloc[0]

for eth in ethnicity_categories:
    ethnicity_actual[eth] = ethnicity_df[eth].iloc[0]

for rel in religion_categories:
    religion_actual[rel] = religion_df[rel].iloc[0]

for mar in marital_categories:
    marital_actual[mar] = marital_df[mar].iloc[0]

# Calculate predicted distributions
sex_pred_counts = dict(Counter(sex_pred_names))
age_pred_counts = dict(Counter(age_pred_names))
ethnicity_pred_counts = dict(Counter(ethnicity_pred_names))
religion_pred_counts = dict(Counter(religion_pred_names))
marital_pred_counts = dict(Counter(marital_pred_names))

# Normalize the actual distributions to match the total number of persons in predictions
# This ensures fair comparison of relative proportions
total_actual_sex = sum(sex_actual.values())
total_actual_age = sum(age_actual.values())
total_actual_ethnicity = sum(ethnicity_actual.values())
total_actual_religion = sum(religion_actual.values())
total_actual_marital = sum(marital_actual.values())
total_pred = num_persons

if total_actual_sex > 0:
    sex_actual = {k: v * total_pred / total_actual_sex for k, v in sex_actual.items()}
if total_actual_age > 0:
    age_actual = {k: v * total_pred / total_actual_age for k, v in age_actual.items()}
if total_actual_ethnicity > 0:
    ethnicity_actual = {k: v * total_pred / total_actual_ethnicity for k, v in ethnicity_actual.items()}
if total_actual_religion > 0:
    religion_actual = {k: v * total_pred / total_actual_religion for k, v in religion_actual.items()}
if total_actual_marital > 0:
    marital_actual = {k: v * total_pred / total_actual_marital for k, v in marital_actual.items()}

# Create combined age-sex column names
age_sex_combinations = [f"{age} {sex}" for age in age_groups for sex in sex_categories]

# Create actual crosstables with ethnicity/religion/marital as indices and age-sex combinations as columns
ethnic_sex_age_actual = pd.DataFrame(0, index=ethnicity_categories, columns=age_sex_combinations)
religion_sex_age_actual = pd.DataFrame(0, index=religion_categories, columns=age_sex_combinations)
marital_sex_age_actual = pd.DataFrame(0, index=marital_categories, columns=age_sex_combinations)

# Extract the actual counts from the crosstable dataframes
for sex in sex_categories:
    for age in age_groups:
        col_name = f"{age} {sex}"
        # Sum up counts for each ethnicity for this sex-age combination
        for eth in ethnicity_categories:
            original_col = f'{sex} {age} {eth}'
            if original_col in ethnic_by_sex_by_age_df.columns:
                ethnic_sex_age_actual.loc[eth, col_name] = ethnic_by_sex_by_age_df[original_col].iloc[0]
        
        # Sum up counts for each religion for this sex-age combination
        for rel in religion_categories:
            original_col = f'{sex} {age} {rel}'
            if original_col in religion_by_sex_by_age_df.columns:
                religion_sex_age_actual.loc[rel, col_name] = religion_by_sex_by_age_df[original_col].iloc[0]
        
        # Sum up counts for each marital status for this sex-age combination
        for mar in marital_categories:
            original_col = f'{sex} {age} {mar}'
            if original_col in marital_by_sex_by_age_df.columns:
                marital_sex_age_actual.loc[mar, col_name] = marital_by_sex_by_age_df[original_col].iloc[0]

# Create predicted crosstables with the same structure
ethnic_sex_age_pred = pd.DataFrame(0, index=ethnicity_categories, columns=age_sex_combinations)
religion_sex_age_pred = pd.DataFrame(0, index=religion_categories, columns=age_sex_combinations)
marital_sex_age_pred = pd.DataFrame(0, index=marital_categories, columns=age_sex_combinations)

# Fill the predicted crosstables based on our model predictions
for i in range(len(sex_pred_names)):
    sex = sex_pred_names[i]
    age = age_pred_names[i]
    eth = ethnicity_pred_names[i]
    rel = religion_pred_names[i]
    mar = marital_pred_names[i]
    
    col_name = f"{age} {sex}"
    ethnic_sex_age_pred.loc[eth, col_name] += 1
    religion_sex_age_pred.loc[rel, col_name] += 1
    marital_sex_age_pred.loc[mar, col_name] += 1

# Plotly version of individual attribute distribution plots
def plotly_attribute_distributions(attribute_dicts, categories_dict, use_log=False, filter_zero_bars=False, max_cols=2):
    """
    Creates Plotly subplots comparing actual vs. predicted distributions for multiple attributes.
    Now includes a geo plot in the top right corner showing the selected area.
    
    Parameters:
    attribute_dicts - Dictionary of attribute names to (actual, predicted) count dictionaries
    categories_dict - Dictionary of attribute names to lists of categories
    use_log - Whether to use log scale for y-axis
    filter_zero_bars - Whether to filter out bars where both actual and predicted are zero
    max_cols - Maximum number of columns in the subplot grid
    """
    attrs = list(attribute_dicts.keys())
    num_plots = len(attrs)
    
    # Add one extra column for the geo plot
    num_cols = min(num_plots, max_cols) + 1
    num_rows = math.ceil(num_plots / (num_cols - 1))  # Exclude geo column from calculation
    
    # Pre-calculate accuracy for each attribute
    accuracy_data = {}
    for attr_name in attrs:
        actual_dict, predicted_dict = attribute_dicts[attr_name]
        categories = categories_dict[attr_name]
        
        # Filter zero bars if requested
        if filter_zero_bars:
            filtered_cats = [
                cat for cat in categories
                if not (actual_dict.get(cat, 0) == 0 and predicted_dict.get(cat, 0) == 0)
            ]
            categories = filtered_cats
        
        # Calculate R² accuracy
        r2 = calculate_r2_accuracy(
            {cat: predicted_dict.get(cat, 0) for cat in categories},
            {cat: actual_dict.get(cat, 0) for cat in categories}
        )
        accuracy_data[attr_name] = r2 * 100.0
    
    # Create subplot specifications
    specs = []
    for row in range(num_rows):
        row_specs = []
        for col in range(num_cols):
            if row == 0 and col == num_cols - 1:  # Top right corner for geo plot
                row_specs.append({"type": "geo"})
            else:
                row_specs.append({"type": "xy"})
        specs.append(row_specs)
    
    # Create complete subplot titles with accuracy information
    subplot_titles = []
    attr_idx = 0
    
    for row in range(num_rows):
        for col in range(num_cols):
            if row == 0 and col == num_cols - 1:  # Geo plot position
                subplot_titles.append("")  # No title for geo plot
            elif row == 1 and col == num_cols - 1:  # Empty last col in 2nd row
                subplot_titles.append("")
            elif attr_idx < len(attrs):  # Main plot positions
                attr_name = attrs[attr_idx]
                accuracy = accuracy_data[attr_name]
                subplot_titles.append(f"{attr_name} - Accuracy:{accuracy:.2f}%")
                attr_idx += 1
            else:  # Empty positions
                subplot_titles.append("")
    
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles,
        specs=specs,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.10,
        vertical_spacing=0.20
    )
    
    # Add attribute distribution plots
    for idx, attr_name in enumerate(attrs):
        row = (idx // (num_cols - 1)) + 1  # Exclude geo column from calculation
        col = (idx % (num_cols - 1)) + 1   # Exclude geo column from calculation
        
        actual_dict, predicted_dict = attribute_dicts[attr_name]
        categories = categories_dict[attr_name]
        
        # Filter zero bars if requested
        if filter_zero_bars:
            filtered_cats = [
                cat for cat in categories
                if not (actual_dict.get(cat, 0) == 0 and predicted_dict.get(cat, 0) == 0)
            ]
            categories = filtered_cats
        
        # Convert to arrays
        actual_counts = np.array([actual_dict.get(cat, 0) for cat in categories])
        predicted_counts = np.array([predicted_dict.get(cat, 0) for cat in categories])
        
        # Optional log transform
        if use_log:
            actual_counts = np.log1p(actual_counts)
            predicted_counts = np.log1p(predicted_counts)
        
        # Add traces
        actual_trace = go.Bar(
            x=categories,
            y=actual_counts,
            name='Actual' if idx == 0 else None,
            marker_color='red',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        predicted_trace = go.Bar(
            x=categories,
            y=predicted_counts,
            name='Predicted' if idx == 0 else None,
            marker_color='blue',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
    
    # Add geo plot in top right corner
    geo_traces, geo_layout = create_geo_plot_trace(selected_area_code, current_dir)
    
    if geo_traces and geo_layout:
        for trace in geo_traces:
            fig.add_trace(trace, row=1, col=num_cols)
        
        # Update geo subplot layout
        fig.update_geos(
            geo_layout,
            row=1, col=num_cols
        )
    
    # Update layout
    fig.update_layout(
        height=300 * num_rows,
        width=450 * num_cols,  # Fixed width
        title_text="Individual Attributes: Actual vs. Predicted",
        showlegend=True,
        plot_bgcolor="white",
        barmode='group',
        margin=dict(l=40, r=40, t=80, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.3,  # Position below geo plot
            xanchor="center", 
            x=0.85,  # Align with geo plot column
            bgcolor='rgba(255,255,255,0.9)'
        )
    )

    fig.update_xaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    fig.update_yaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    # Display the plot
    fig.show()

# Plotly version of crosstable plots
def plotly_crosstable_comparison(
    actual_dfs, 
    predicted_dfs, 
    titles, 
    show_keys=False, 
    num_cols=1, 
    filter_zero_bars=True
):
    """
    Creates Plotly subplots comparing actual vs. predicted distributions for crosstables.
    Now includes a geo plot in the top right corner showing the selected area.
    
    Parameters:
    actual_dfs - Dictionary of crosstable names to actual dataframes
    predicted_dfs - Dictionary of crosstable names to predicted dataframes
    titles - List of subplot titles
    show_keys - Whether to show full category key combinations (True) or numeric indices (False)
    num_cols - Number of columns in the subplot grid
    filter_zero_bars - Whether to filter out bars where both actual and predicted are zero
    """
    keys_list = list(actual_dfs.keys())
    num_plots = len(keys_list)
    
    # Pre-calculate accuracy for each crosstable
    accuracy_data = {}
    for idx, crosstable_key in enumerate(keys_list):
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays for bar charts
        actual_vals = []
        predicted_vals = []
        
        for i, row_idx in enumerate(actual_df.index):
            for j, col_idx in enumerate(actual_df.columns):
                a_val = actual_df.iloc[i, j]
                p_val = predicted_df.iloc[i, j]
                
                if not filter_zero_bars or not (a_val == 0 and p_val == 0):
                    actual_vals.append(a_val)
                    predicted_vals.append(p_val)
        
        # Calculate R² accuracy using the same method as training
        r2_accuracy = calculate_r2_accuracy(
            {i: predicted_vals[i] for i in range(len(predicted_vals))},
            {i: actual_vals[i] for i in range(len(actual_vals))}
        )
        accuracy_data[idx] = r2_accuracy * 100.0
    
    # Add one extra column for the geo plot
    total_cols = num_cols + 1
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    # Create subplot specifications
    specs = []
    for row in range(num_rows):
        row_specs = []
        for col in range(total_cols):
            if row == 0 and col == total_cols - 1:  # Top right corner for geo plot
                row_specs.append({"type": "geo"})
            else:
                row_specs.append({"type": "xy"})
        specs.append(row_specs)
    
    vertical_spacing = 0.4 if show_keys else 0.2
    subplot_height = 400 if show_keys else 300
    
    if num_rows > 1:
        max_spacing = 1.0 / (num_rows - 1) - 0.01  # subtract a small margin
        vertical_spacing = min(vertical_spacing, max_spacing)
    else:
        vertical_spacing = 0
    
    # Create complete titles with accuracy information
    all_titles = []
    main_plot_idx = 0  # Separate counter for main plots
    
    for i in range(num_rows):
        for j in range(total_cols):
            if j < num_cols:
                # This is a main plot column
                if main_plot_idx < len(titles):
                    accuracy = accuracy_data[main_plot_idx]
                    all_titles.append(f"{titles[main_plot_idx]} - Accuracy:{accuracy:.2f}%")
                    main_plot_idx += 1
                else:
                    all_titles.append("")
            elif i == 0 and j == total_cols - 1:
                # Geo plot title (first row, last column) - removed title
                all_titles.append("")
            else:
                # Empty titles for other areas
                all_titles.append("")
    
    fig = make_subplots(
        rows=num_rows,
        cols=total_cols,
        subplot_titles=all_titles,
        specs=specs,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.08,
        column_widths=[0.9] * num_cols + [0.1]  # 90% main plots, 10% geo plot
    )
    
    for idx, crosstable_key in enumerate(keys_list):
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1
        
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays for bar charts
        actual_vals = []
        predicted_vals = []
        original_indices = []  # Store original indices from glossary
        
        sequential_index = 1  # Start from 1 to match glossary numbering
        for i, row_idx in enumerate(actual_df.index):
            for j, col_idx in enumerate(actual_df.columns):
                a_val = actual_df.iloc[i, j]
                p_val = predicted_df.iloc[i, j]
                
                if not filter_zero_bars or not (a_val == 0 and p_val == 0):
                    actual_vals.append(a_val)
                    predicted_vals.append(p_val)
                    original_indices.append(sequential_index)  # Keep original glossary index
                
                sequential_index += 1  # Always increment to maintain glossary alignment
        
        # Use original indices as x-axis labels
        x_labels = original_indices
        
        # Create continuous positions for bars (no gaps) but keep original indices as labels
        continuous_positions = list(range(1, len(actual_vals) + 1))
        original_indices_labels = original_indices
        
        # Calculate label step size based on number of bars
        # Show fewer labels if there are many bars
        num_bars = len(continuous_positions)
        if num_bars > 300:
            step_size = 3
        elif num_bars > 100:
            step_size = 2
        else:
            step_size = 1
            
        # Create visible labels array with appropriate step size
        visible_labels = []
        visible_positions = []
        for i, (pos, label) in enumerate(zip(continuous_positions, original_indices_labels)):
            if i % step_size == 0:
                visible_labels.append(str(label))
                visible_positions.append(pos)
            else:
                visible_labels.append("")
                visible_positions.append(pos)
        
        # Create bar traces using continuous positions
        actual_trace = go.Bar(
            x=continuous_positions,
            y=actual_vals,
            name='Actual' if idx == 0 else None,
            marker_color='red',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        predicted_trace = go.Bar(
            x=continuous_positions,
            y=predicted_vals,
            name='Predicted' if idx == 0 else None,
            marker_color='blue',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
        
        # Update x-axis to show original indices as labels at continuous positions
        fig.update_xaxes(
            ticktext=visible_labels,
            tickvals=visible_positions,
            tickangle=90,  # Angle the labels for better readability
            row=row,
            col=col
        )
    
    # Add geo plot in top right corner
    geo_traces, geo_layout = create_geo_plot_trace(selected_area_code, current_dir)
    
    if geo_traces and geo_layout:
        for trace in geo_traces:
            fig.add_trace(trace, row=1, col=total_cols)
        
        # Update geo subplot layout
        fig.update_geos(
            geo_layout,
            row=1, col=total_cols
        )
    
    # Update layout to be responsive and dynamic
    fig.update_layout(
        height=subplot_height * num_rows + 100,  # Add extra space for titles and margins
        title_text="Crosstable Comparison: Actual vs. Predicted",
        showlegend=True,
        barmode='group',
        plot_bgcolor="white",
        autosize=True,  # Enable responsive sizing
        margin=dict(
            b=200 if show_keys else 100,
            t=100,
            l=40,
            r=40
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.3,  # Position below geo plot
            xanchor="center", 
            x=0.95,  # Align with geo plot column (adjusted for 90/10 layout)
            bgcolor='rgba(255,255,255,0.9)'
        )
    )
    
    fig.update_yaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    # Add x-axis line styling without overriding tick settings
    fig.update_xaxes(
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    # Display the plot
    fig.show()

# Prepare data for visualization
# Create attribute dictionaries for plotting
attribute_dicts = {
    'Sex': (sex_actual, sex_pred_counts),
    'Age': (age_actual, age_pred_counts),
    'Ethnicity': (ethnicity_actual, ethnicity_pred_counts),
    'Religion': (religion_actual, religion_pred_counts),
    'Marital Status': (marital_actual, marital_pred_counts)
}

categories_dict = {
    'Sex': sex_categories,
    'Age': age_groups,
    'Ethnicity': ethnicity_categories,
    'Religion': religion_categories,
    'Marital Status': marital_categories
}

# Plot individual attribute distributions
plotly_attribute_distributions(attribute_dicts, categories_dict, filter_zero_bars=True)

# Create crosstables for visualization
# Create crosstable dictionaries for plotting
actual_dfs = {
    'Ethnic_Sex_Age': ethnic_sex_age_actual,
    'Religion_Sex_Age': religion_sex_age_actual,
    'Marital_Sex_Age': marital_sex_age_actual
}

# print("============ Actual Crosstables DF =============")
# print(actual_dfs)

predicted_dfs = {
    'Ethnic_Sex_Age': ethnic_sex_age_pred,
    'Religion_Sex_Age': religion_sex_age_pred,
    'Marital_Sex_Age': marital_sex_age_pred
}

# print("============ Predicted Crosstables DF =============")
# print(predicted_dfs)

titles = [
    'Ethnicity x Sex x Age',
    'Religion x Sex x Age',
    'Marital Status x Sex x Age'
]

# Plot crosstable comparisons using the same function as in generateHouseholds.py
plotly_crosstable_comparison(actual_dfs, predicted_dfs, titles, show_keys=False, filter_zero_bars=True)

# Plotly version of radar crosstable comparison
def plotly_radar_crosstable_comparison(actual_dfs, predicted_dfs, titles):
    """
    Creates radar chart subplots comparing actual vs. predicted distributions for crosstables.
    Uses numeric indices instead of category labels and shows aggregated actual vs predicted lines.
    Now includes a geo plot showing the selected area.
    
    Parameters:
    actual_dfs - Dictionary of crosstable names to actual dataframes
    predicted_dfs - Dictionary of crosstable names to predicted dataframes
    titles - List of subplot titles
    """
    keys_list = list(actual_dfs.keys())
    num_plots = len(keys_list)
    
    # Pre-calculate accuracy for each crosstable
    accuracy_data = {}
    for idx, crosstable_key in enumerate(keys_list):
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays
        actual_vals = actual_df.values.flatten()
        predicted_vals = predicted_df.values.flatten()
        
        # Calculate R² accuracy using the same method as training
        r2_accuracy = calculate_r2_accuracy(
            {i: predicted_vals[i] for i in range(len(predicted_vals))},
            {i: actual_vals[i] for i in range(len(actual_vals))}
        )
        accuracy_data[idx] = r2_accuracy * 100.0
    
    # Set to two columns: one for radar charts, one for geo plot
    num_cols = 2
    num_rows = num_plots
    
    # Create subplot specifications
    specs = []
    for row in range(num_rows):
        row_specs = [{'type': 'polar'}]  # Radar chart column
        if row == 0:  # Only add geo to first row
            row_specs.append({'type': 'geo'})  # Geo plot column
        else:
            row_specs.append(None)  # Empty subplot for other rows
        specs.append(row_specs)
    
    # Create complete titles with accuracy information
    extended_titles = []
    for i, title in enumerate(titles):
        accuracy = accuracy_data[i]
        extended_titles.append(f"{title} - Accuracy:{accuracy:.2f}%")
        
        if i == 0:  # Add geo title only for first row
            extended_titles.append("")  # Removed geo plot title
        else:
            extended_titles.append("")  # Empty title for other rows
    
    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=extended_titles,
        specs=specs,
        vertical_spacing=0.1,  # Increased vertical spacing between subplots
        horizontal_spacing=0.2,
        column_widths=[0.7, 0.3]  # 70% radar charts, 30% geo plot
    )
    
    for idx, crosstable_key in enumerate(keys_list):
        row = idx + 1
        col = 1  # Always put radar charts in first column
        
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays
        actual_vals = actual_df.values.flatten()
        predicted_vals = predicted_df.values.flatten()
        
        # Create numeric indices for the categories
        num_points = len(actual_vals)
        
        # Determine step size for labels based on number of points
        if num_points > 400:
            step_size = 16
        elif num_points > 200:
            step_size = 8
        elif num_points > 40:
            step_size = 4
        elif num_points > 30:
            step_size = 3
        elif num_points > 20:
            step_size = 2
        else:
            step_size = 1
            
        # Create labels with appropriate step size
        theta = []
        for i in range(num_points):
            if i % step_size == 0:
                theta.append(f"{i+1}")
            else:
                theta.append("")
        
        # Add the first value again to close the polygon
        actual_vals = np.append(actual_vals, actual_vals[0])
        predicted_vals = np.append(predicted_vals, predicted_vals[0])
        theta = theta + [theta[0]]
        
        # Get pre-calculated accuracy
        r2_accuracy = accuracy_data[idx]
        
        # Create traces
        actual_trace = go.Scatterpolar(
            r=actual_vals,
            theta=theta,
            name='Actual' if idx == 0 else None,
            line=dict(color='red', width=2),
            showlegend=idx == 0
        )
        
        predicted_trace = go.Scatterpolar(
            r=predicted_vals,
            theta=theta,
            name=f'Predicted (Acc: {r2_accuracy:.1f}%)' if idx == 0 else None,
            line=dict(color='blue', width=2),
            showlegend=idx == 0
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
    
    # Add geo plot in top right (first row, second column)
    geo_traces, geo_layout = create_geo_plot_trace(selected_area_code, current_dir)
    
    if geo_traces and geo_layout:
        for trace in geo_traces:
            fig.add_trace(trace, row=1, col=2)
        
        # Update geo subplot layout
        fig.update_geos(
            geo_layout,
            row=1, col=2
        )
    
    # Update layout with fixed dimensions
    fig.update_layout(
        height=450 * num_rows,  # Slightly increased height to accommodate title spacing
        width=1200,  # Fixed width
        title_text="Radar Chart Comparison: Actual vs. Predicted",
        title_font_size=18,  # Main title size
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.3,  # Position below geo plot
            xanchor="center", 
            x=0.85,  # Align with geo plot column (for 70/30 layout)
            bgcolor='rgba(255,255,255,0.9)'
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(
                    actual_df.values.max(),
                    predicted_df.values.max()
                )]
            )
        ),
        margin=dict(t=120, b=80, l=100, r=100)  # Increased top margin for titles
    )
    
    # Update polar axes for each subplot
    for i in range(1, num_rows + 1):
        fig.update_polars(
            dict(
                radialaxis=dict(
                    visible=True,
                    showline=True,
                    showticklabels=True,
                    gridcolor="lightgrey",
                    gridwidth=0.5,
                    tickfont=dict(size=8),  # Reduced radial axis font size
                ),
                angularaxis=dict(
                    showline=True,
                    showticklabels=True,
                    gridcolor="lightgrey",
                    gridwidth=0.5,
                    tickfont=dict(size=8),  # Reduced angular axis font size
                    rotation=90,
                    direction="clockwise"
                )
            ),
            row=i,
            col=1  # Only apply to radar chart column
        )
    
    # Display the plot
    fig.show()

# Plot radar chart comparisons
# plotly_radar_crosstable_comparison(actual_dfs, predicted_dfs, titles)