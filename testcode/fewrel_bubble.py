import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 示例数据
data = {
    'Dataset': ['Example'] * 5,
    'Model': ['ZETT', 'PCRED', 'ZS_SKA','AML_EIKC','KBPT'],
    'CFA': [3.53, 4.85, 1.78, 10.79, 1.87]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置绘图风格
sns.set(style="ticks")

# 创建FacetGrid
g = sns.FacetGrid(df, hue='Model',  margin_titles=True, height=4, aspect=1)

# 绘制散点图
g.map(sns.stripplot, 'Model', 'CFA', dodge=True, jitter=True, alpha=0.7, size=10, edgecolor='w',linewidth=1)

# 添加图例
g.add_legend(fontsize=14)

# 调整布局
plt.subplots_adjust(top=0.9)

g.set_axis_labels('Model', 'CFA', fontsize=14)
plt.xticks([])
plt.show()