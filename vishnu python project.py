import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Load data
file_path = r"/Users/vishnu/Desktop/S4/INT375 PYTHON/CA2 Project/PROJECT.csv"
df = pd.read_csv(file_path)

print(df.info())
print(df.describe())

# Handle Missing Values 
df['pollutant_id'] = df['pollutant_id'].fillna(df['pollutant_id'].mode()[0])
df['pollutant_avg'] = df['pollutant_avg'].fillna(df['pollutant_avg'].mean())
df['pollutant_min'] = df['pollutant_min'].fillna(df['pollutant_min'].mean())
df['pollutant_max'] = df['pollutant_max'].fillna(df['pollutant_max'].mean())
df['station'] = df['station'].fillna(df['station'].mode()[0])
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')

# Set attractive dark theme
sb.set_theme(style="darkgrid", font_scale=1.2, rc={"font.family": "Arial", "axes.facecolor": "#2c3e50", "figure.facecolor": "#2c3e50"})

# Objective 1: Pollution Pulse – Checking the Pulse of Pollution Levels
plt.figure(figsize=(12, 8))
sb.violinplot(x="state", y="pollutant_avg", hue="state", data=df, palette="viridis", inner="quartile", legend=False)
plt.title('Pollution Pulse: Distribution Across States', fontsize=16, weight='bold', color='white')
plt.xlabel('State', fontsize=14, color='white')
plt.ylabel('Average Pollutant Level', fontsize=14, color='white')
plt.xticks(rotation=45, ha='right', color='white')
plt.yticks(color='white')
# Add mean annotations
means = df.groupby('state')['pollutant_avg'].mean()
for i, mean in enumerate(means):
    plt.text(i, mean, f'{mean:.1f}', ha='center', va='bottom', color='white', fontsize=10)
plt.tight_layout()
plt.savefig('pollution_pulse.png', dpi=300, bbox_inches='tight')
plt.show()

# Objective 2:Correlation Between Pollutants
plt.figure(figsize=(10, 8))
correlation_matrix = df[['pollutant_min', 'pollutant_max', 'pollutant_avg']].corr()
sb.heatmap(correlation_matrix, annot=True, cmap="RdBu", vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .5})
plt.title('Correlation Between Pollutants', fontsize=16, weight='bold', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.savefig('correlation_pollutants.png', dpi=300, bbox_inches='tight')
plt.show()





# Set theme with blue background
plt.style.use('default')  # Reset to default to avoid darkgrid interference
plt.rcParams['figure.facecolor'] = '#4682B4'  # Steel Blue background
plt.rcParams['axes.facecolor'] = '#4682B4'    # Match axes background
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

geo_data = df.groupby(['city', 'latitude', 'longitude'])['pollutant_avg'].mean().reset_index()

# Objective 3: Plot scatter map
plt.figure(figsize=(12, 8))
plt.scatter(geo_data['longitude'], geo_data['latitude'], 
            s=geo_data['pollutant_avg']*10,  # Scale bubble size
            c=geo_data['pollutant_avg'], cmap='Greens', alpha=0.6)
plt.colorbar(label='Average Pollutant Level')
plt.title('Pollution Intensity Across Cities', fontsize=16, weight='bold', color='white')
plt.xlabel('Longitude', fontsize=14, color='white')
plt.ylabel('Latitude', fontsize=14, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.savefig('pollution_intensity.png', dpi=300, bbox_inches='tight')
plt.show()

#Objective 4: Top and Bottom Cities by Pollution Level (Sorted Column Chart)
# Group by city and calculate average pollutant level
city_avg = df.groupby('city')['pollutant_avg'].mean().sort_values()

# Get top 10 and bottom 10 cities
top_10 = city_avg.tail(10)
bottom_10 = city_avg.head(10)

# Combine top and bottom
combined = pd.concat([bottom_10, top_10])

# Plot sorted column chart
plt.figure(figsize=(12, 6))
combined.plot(kind='bar', color=['green']*10 + ['red']*10)
plt.title('Top 10 and Bottom 10 Cities by Average Pollution Level')
plt.xlabel('City')
plt.ylabel('Average Pollutant Level')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()





# Objective 5: Pollution Level Comparison by State
# Prepare data for top 3 states
top_states = df.groupby('state')['pollutant_avg'].mean().nlargest(3).index
df_top = df[df['state'].isin(top_states)].copy()

# Sort by pollutant_max for smoother line plot
df_top = df_top.sort_values('pollutant_max')

# Fit linear regression and calculate R² for each state
predictions = {}
r2_scores = {}
for state in top_states:
    state_data = df_top[df_top['state'] == state]
    X = state_data[['pollutant_max']]  # DataFrame with feature names
    y = state_data['pollutant_avg']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    r2_scores[state] = r2
    # Create X_range as a DataFrame with the same column name
    X_range = pd.DataFrame([[min(X['pollutant_max'])], [max(X['pollutant_max'])]], columns=['pollutant_max'])
    y_pred_range = model.predict(X_range)
    predictions[state] = y_pred_range

# Define dark color palette
dark_colors = ['#8B0000', '#006400', '#00008B']  # Dark Red, Dark Green, Dark Blue

# Plot actual data and regression lines
plt.figure(figsize=(12, 6))
sb.lineplot(x='pollutant_max', y='pollutant_avg', hue='state', data=df_top, palette=dark_colors, alpha=0.5, legend='brief')
for state, pred in predictions.items():
    x_range = [min(df_top['pollutant_max']), max(df_top['pollutant_max'])]
    plt.plot(x_range, pred, label=f'Regression ({state}, R²={r2_scores[state]:.2f})', 
             color=dark_colors[list(top_states).index(state)], linewidth=2)
plt.title('Pollution Level Comparison by State', fontsize=16, weight='bold', color='white')
plt.xlabel('Maximum Pollutant Level', fontsize=14, color='white')
plt.ylabel('Average Pollutant Level', fontsize=14, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.legend(title='State/Regression', title_fontsize=12, labelcolor='white', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, color='white', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('pollution_level_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print R² scores for reference
for state, r2 in r2_scores.items():
    print(f'R² Score for {state}: {r2:.4f}')
