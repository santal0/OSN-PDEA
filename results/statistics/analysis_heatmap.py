import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']

file_path = 'all_steps_shopping.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Missing file: {file_path}")

df = pd.read_csv(file_path)
dims = ['I', 'F', 'P', 'C', 'S', 'N', 'FA']

profile_mapping = {
    'bargain_hunter': 'Bargain Hunter',
    'high_payment_shopping': 'High Payment',
    'low_payment_browsing': 'Low Payment'
}
df['profile_id'] = df['profile_id'].map(profile_mapping)
df['app'] = df['app'].str.upper()

profiles = ['Bargain Hunter', 'High Payment', 'Low Payment']

fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

for i, prof in enumerate(profiles):
    prof_data = df[df['profile_id'] == prof]
    mean_matrix = prof_data.groupby('app')[dims].mean().T
    
    sns.heatmap(
        mean_matrix, 
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0, 
        vmax=1,
        cbar=False,
        linewidths=1.5,
        linecolor='white',
        annot_kws={"size": 12, "weight": "bold"},
        ax=axes[i]
    )
    
    axes[i].set_title(f"{prof} Profile", fontsize=16, fontweight='bold', pad=15)
    axes[i].set_xlabel("E-Commerce Platforms", fontsize=12, labelpad=10)
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=13)

axes[0].set_ylabel("DPS Dimensions", fontsize=14, fontweight='bold', labelpad=10)

mappable = axes[0].collections[0]
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
cbar = fig.colorbar(mappable, cax=cbar_ax)
cbar.set_label('Dark Pattern Exposure Rate', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=11)

plt.suptitle("Differential Dark Pattern Exposure Matrix Across Profiles & Platforms", 
             fontsize=20, fontweight='bold', y=0.98)

plt.subplots_adjust(left=0.08, right=0.91, top=0.85, wspace=0.15)

output_name = 'heatmap_shopping.png'
plt.savefig(output_name, dpi=300, bbox_inches='tight')
plt.close()
print(f"Success! Heatmap chart saved as '{output_name}'.")