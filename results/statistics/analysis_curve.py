import os
import matplotlib.pyplot as plt
import pandas as pd

csv_path = 'all_steps_shopping.csv'

df = pd.read_csv(csv_path)

DPS_DIMS = ["I", "F", "P", "C", "S", "N", "FA"]

output_dir = 'curves_shopping'
os.makedirs(output_dir, exist_ok=True)

profiles = sorted(df['profile_id'].unique())
apps = sorted(df['app'].unique())

print("Start working...")

for pid in profiles:
    for app in apps:
        sub_df = df[(df['profile_id'] == pid) & (df['app'] == app)].sort_values('step')
        
        if sub_df.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        
        xs = list(range(1, len(sub_df) + 1))
        
        for dim in DPS_DIMS:
            dim_values = sub_df[dim].astype(float).values
            vals = []
            acc = 0.0
            for idx, val in enumerate(dim_values, start=1):
                acc += val
                vals.append(acc / idx)

            plt.plot(xs, vals, marker="o", label=dim)
            
        plt.title(f"Running E(D_i | A={pid}) - [{app.upper()}]")
        plt.xlabel("Step")
        plt.ylabel("Running mean exposure")
        plt.legend()
        plt.tight_layout()
        
        filename = f"curve_E_{pid}_{app}_shopping.png"
        save_path = os.path.join(output_dir, filename)
        
        # 保存并关闭画布
        plt.savefig(save_path, dpi=160)
        plt.close()

print(f"Success!:'{output_dir}/'")