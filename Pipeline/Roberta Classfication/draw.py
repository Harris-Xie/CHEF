import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# 定义数据
length_id = [0, 1, 2, 3, 4]
f1_micro = [85.00, 83.64, 69.83, 80.00, 79.53]
precision_macro = [85.69, 86.89, 68.33, 82.22, 84.92]
recall_macro = [79.60, 82.01, 75.81, 72.33, 79.22]
f1_macro = [79.97, 83.03, 60.62, 75.51, 79.18]

# 创建DataFrame
data = {
    'Length ID': length_id,
    'F1 (micro)': f1_micro,
    'Precision (macro)': precision_macro,
    'Recall (macro)': recall_macro,
    'F1 (macro)': f1_macro
}
df = pd.DataFrame(data)

# 绘制柱状图
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# 设置调色板和颜色
color_palette = sns.color_palette("Set2")
sns.barplot(x="Length ID", y="Value", hue="Metric", data=df.melt('Length ID', var_name='Metric', value_name='Value'), palette=color_palette)

# 添加标题和轴标签
plt.title("Metrics by Length ID", fontsize=16)
plt.xlabel("Length ID", fontsize=12)
plt.ylabel("Value", fontsize=12)

# 调整刻度标签字体和字号
plt.xticks(ticks=df['Length ID'], labels=['Politics', 'Society', 'Culture', 'Science', 'Public Health'],fontsize=10)
plt.yticks(fontsize=10)

# 添加图例

plt.legend(title="Metric", fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('barplot.png')
plt.show()