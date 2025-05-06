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

# Device selection with better fallback options
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                      'cpu')
print(f"Using device: {device}")

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
age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/New-Tables/individual/Age_Perfect_5yrs.csv'))
sex_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/New-Tables/individual/Sex.csv'))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/New-Tables/individual/Ethnicity.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/New-Tables/individual/Religion.csv'))
marital_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/New-Tables/individual/Marital.csv'))
# age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Age_5yrs.csv'))
# sex_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Sex.csv'))
# ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Ethnic.csv'))
# religion_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Religion.csv'))
# marital_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Marital.csv'))
ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/New-Tables/crosstables/EthnicityBySexByAge.csv'))
religion_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/New-Tables/crosstables/ReligionbySexbyAge.csv'))
marital_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/New-Tables/crosstables/MaritalbySexbyAgeModified.csv'))
# ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/ethnic_by_sex_by_age_modified.csv'))
# religion_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/religion_by_sex_by_age.csv'))
# marital_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/marital_by_sex_by_age.csv'))

# Define the Oxford areas
oxford_areas = ['E02005924']
# oxford_areas = ['E02005923']
# oxford_areas = ['E02005925']

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
# age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_44', '45_59', '60_64', '65_74', '75_84', '85_89', '90+']
sex_categories = ['M', 'F']
# ethnicity_categories = ['W0', 'M0', 'A0', 'B0', 'O0']
# religion_categories = ['C','B','H','J','M','S','OR','NR','NS']
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
        self.dropout = torch.nn.Dropout(0.5)
        
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

# Define the hyperparameters to tune
learning_rates = [0.001, 0.0005, 0.0001]
# learning_rates = [0.001]
# hidden_channel_options = [64]
hidden_channel_options = [64, 128, 256]
mlp_hidden_dim = 128
num_epochs = 2500
# num_epochs = 10

# Results storage
results = []

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
        for i in range(len(targets)):
            loss += custom_loss_function(
                out[targets[i][0][0]], out[targets[i][0][1]], out[targets[i][0][2]],
                targets[i][1][0], targets[i][1][1], targets[i][1][2]
            )

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print metrics for every epoch since num_epochs is small now
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluate accuracy after training
    with torch.no_grad():
        age_out, sex_out, ethnicity_out, religion_out, marital_out = model(data)
        age_pred = age_out[:num_persons].argmax(dim=1)
        sex_pred = sex_out[:num_persons].argmax(dim=1)
        ethnicity_pred = ethnicity_out[:num_persons].argmax(dim=1)
        religion_pred = religion_out[:num_persons].argmax(dim=1)
        marital_pred = marital_out[:num_persons].argmax(dim=1)

        # Calculate net accuracy across all tasks
        net_accuracy = 0
        for i in range(len(targets)):
            pred_1 = out[targets[i][0][0]].argmax(dim=1)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)
            pred_3 = out[targets[i][0][2]].argmax(dim=1)
            task_net_accuracy = ((pred_1 == targets[i][1][0]) & (pred_2 == targets[i][1][1]) & (pred_3 == targets[i][1][2])).sum().item() / num_persons
            net_accuracy += task_net_accuracy

        # Return the final loss and averaged net accuracy
        return loss.item(), net_accuracy / len(targets), (sex_pred, age_pred, ethnicity_pred, religion_pred, marital_pred)

# Run the grid search over hyperparameters
total_start_time = time.time()
time_results = []

for lr in learning_rates:
    for hidden_channels in hidden_channel_options:
        print(f"Training with lr={lr}, hidden_channels={hidden_channels}")
        
        # Start timing for this combination
        start_time = time.time()
        
        # Train the model for the current combination of hyperparameters
        final_loss, avg_accuracy, predictions = train_model(lr, hidden_channels, num_epochs, data, targets)
        
        # End timing for this combination
        end_time = time.time()
        train_time = end_time - start_time
        train_time_str = str(timedelta(seconds=int(train_time)))
        
        # Store the results
        results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'final_loss': final_loss,
            'average_accuracy': avg_accuracy,
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
        print(f"Final Loss: {final_loss}, Average Accuracy: {avg_accuracy}")
        print(f"Training time: {train_time_str}")

# Calculate total training time
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
total_training_time_str = str(timedelta(seconds=int(total_training_time)))
print(f"Total training time: {total_training_time_str}")

# After all runs, display results
results_df = pd.DataFrame(results)
print("Hyperparameter tuning results:")
print(results_df)

# Add total time as a summary row
total_row = pd.DataFrame([{
    'learning_rate': 'Total',
    'hidden_channels': '',
    'final_loss': '',
    'average_accuracy': '',
    'training_time': total_training_time_str
}])
summary_df = pd.concat([results_df, total_row], ignore_index=True)

# Create a DataFrame for timing results
time_results_df = pd.DataFrame(time_results)

# Save the results to CSV files for future reference
output_dir = os.path.join(current_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'generateIndividuals_hyperparameter_tuning_results.csv')
summary_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# Create detailed timing report as text file
with open(os.path.join(output_dir, 'training_time_results.txt'), 'w') as f:
    f.write("Hyperparameter Training Time Results\n")
    f.write("==================================\n\n")
    
    for i, result in enumerate(time_results):
        f.write(f"Combination {i+1}:\n")
        f.write(f"  Learning Rate: {result['learning_rate']}\n")
        f.write(f"  Hidden Channels: {result['hidden_channels']}\n")
        f.write(f"  Training Time: {result['training_time']}\n")
        f.write("\n")
    
    f.write("==================================\n")
    f.write(f"Total Training Time: {total_training_time_str}\n")
    f.write("==================================\n\n")
    
    f.write("Final Results Table:\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")
    
    # Add device information
    f.write(f"Training Device: {device}\n")
    f.write(f"Number of Epochs: {num_epochs}\n")

print("Training time results saved to training_time_results.txt")

# Extract the best model's predictions (the last one for simplicity, but in practice would pick based on best performance)
sex_pred, age_pred, ethnicity_pred, religion_pred, marital_pred = predictions

# Create person tensor with attributes
# Format: [person_id, age, religion, ethnicity, marital, sex]
person_tensor = torch.zeros(num_persons, 6, device=device)
person_tensor[:, 0] = torch.arange(num_persons, device=device)  # Person ID
person_tensor[:, 1] = age_pred  # Age
person_tensor[:, 2] = religion_pred  # Religion
person_tensor[:, 3] = ethnicity_pred  # Ethnicity
person_tensor[:, 4] = marital_pred  # Marital status
person_tensor[:, 5] = sex_pred  # Sex

# Create output directory if it doesn't exist
output_dir = os.path.join(current_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Save person tensor - move to CPU for saving
person_tensor_path = os.path.join(output_dir, 'person_nodes.pt')
torch.save(person_tensor.cpu(), person_tensor_path)
print(f"Person tensor saved to {person_tensor_path}")