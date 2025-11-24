import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== 1. 读取数据 =====
# 替换成你自己的文件名
df_max = pd.read_csv("results_feasibility2.csv")
df_heu = pd.read_csv("results_heuristic2.csv")

# 计算满足率
for df in (df_max, df_heu):
    df["satisfaction_ratio"] = df["max_flow_value"] / df["total_demand"]

df_max["algo_name"] = "MaxFlow"
df_heu["algo_name"] = "Heuristic"

# ===== 2. 设置输出文件夹 =====
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)  # 若不存在则创建

# ===== 3. 修复后的辅助函数 =====
def subset_by_tag_prefix(df, prefix, new_col, as_float=True):
    sub = df[df["tag"].astype(str).str.startswith(prefix, na=False)].copy()
    sub[new_col] = sub["tag"].astype(str).str.split("=", n=1).str[-1]
    sub[new_col] = sub[new_col].astype(float if as_float else int)
    sub = sub.drop_duplicates(subset=[new_col])
    sub = sub.sort_values(new_col)
    return sub


# ===== 4. 通用绘图函数 =====
def plot_param_compare(
    df_max,
    df_heu,
    prefix,
    x_col_name,
    x_label,
    title,
    save_name,
    as_float=True,
):
    sub_max = subset_by_tag_prefix(df_max, prefix, x_col_name, as_float)
    sub_heu = subset_by_tag_prefix(df_heu, prefix, x_col_name, as_float)

    if sub_max.empty or sub_heu.empty:
        print(f"⚠️ Skip {prefix}: no data in one of the files.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(
        sub_max[x_col_name],
        sub_max["satisfaction_ratio"],
        marker="o",
        label="MaxFlow",
        linewidth=2,
    )
    plt.plot(
        sub_heu[x_col_name],
        sub_heu["satisfaction_ratio"],
        marker="s",
        label="Heuristic",
        linewidth=2,
    )
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel("Satisfaction ratio (max_flow / total_demand)", fontsize=11)
    plt.title(title, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===== 5. 调用绘图函数生成所有图 =====
plot_param_compare(
    df_max,
    df_heu,
    prefix="tau_b=",
    x_col_name="tau_b",
    x_label="Battery capacity τ_b (seconds)",
    title="Effect of Battery Capacity on Satisfaction Ratio",
    save_name="battery_capacity.png",
    as_float=False,
)

plot_param_compare(
    df_max,
    df_heu,
    prefix="P_solar=",
    x_col_name="P_solar",
    x_label="Solar panel power (W)",
    title="Effect of Solar Panel Power on Satisfaction Ratio",
    save_name="solar_power.png",
    as_float=True,
)

plot_param_compare(
    df_max,
    df_heu,
    prefix="exec=",
    x_col_name="exec_time",
    x_label="Average execution time per job (s)",
    title="Effect of Execution Time on Satisfaction Ratio",
    save_name="execution_time.png",
    as_float=False,
)

plot_param_compare(
    df_max,
    df_heu,
    prefix="period=",
    x_col_name="period",
    x_label="Average task period (s)",
    title="Effect of Task Period on Satisfaction Ratio",
    save_name="task_period.png",
    as_float=False,
)

print(f"\n🎉 所有图像已保存在文件夹：{output_dir}/")