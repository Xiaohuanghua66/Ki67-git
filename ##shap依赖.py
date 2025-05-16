##shap依赖
import shap
# 创建解释器
explainer = shap.TreeExplainer(best_model)
# 计算 SHAP 值
shap_values = explainer.shap_values(X_test)

shap.dependence_plot('Resnet101_ROI', shap_values, X_test, interaction_index=None, show=False)
plt.savefig("1.pdf", format='pdf', bbox_inches='tight', dpi=1200)
shap.dependence_plot('INTRAPERI_0-10mm', shap_values, X_test, interaction_index=None, show=False)
plt.savefig("radio.pdf", format='pdf', bbox_inches='tight', dpi=1200)
# 绘制 SHAP 依赖图，指定交互特征
shap.dependence_plot('Resnet101_ROI', shap_values, X_test, interaction_index='INTRAPERI_0-10mm', show=False)
plt.savefig("交互2.pdf", format='pdf', bbox_inches='tight', dpi=1200)
shap_interaction_values = explainer.shap_interaction_values(X_test)  
print("SHAP 交互值维度:",shap_interaction_values.shape)
shap.summary_plot(shap_interaction_values, X_test, show=False)
plt.savefig("4.pdf", format='pdf', bbox_inches='tight', dpi=1200)
# 使用 X_test.columns 作为特征名称
feature_names = X_test.columns

# 初始化一个字典用于存储 DataFrame
dataframes = {}

# 遍历每个特征
for i, feature in enumerate(feature_names):
    # 初始化一个 DataFrame，用于存储当前特征与其他特征的交互值
    interaction_dict = {f"{other_feature}": [] for j, other_feature in enumerate(feature_names) if j != i}
    
    # 遍历每个样本，提取当前特征与其他特征的交互值
    for sample_idx in range(shap_interaction_values.shape[0]):
        # 当前样本的交互矩阵
        interaction_matrix = shap_interaction_values[sample_idx]
        
        # 提取当前特征与其他特征的交互值
        interactions = [interaction_matrix[i, j] for j in range(len(feature_names)) if j != i]
        
        # 添加到对应列
        for col_idx, col_name in enumerate(interaction_dict.keys()):
            interaction_dict[col_name].append(interactions[col_idx])
    
    # 创建 DataFrame
    df = pd.DataFrame(interaction_dict)
    
    # 修改列名为 df_<当前特征名>
    df.columns = [f"df_{feature}_{col}" for col in interaction_dict.keys()]
    
    # 存储到字典
    dataframes[f"df_{i + 1}"] = df

# 将 DataFrame 转为全局变量
for i, (name, df) in enumerate(dataframes.items()):
    globals()[f"df_{i + 1}"] = df  # 动态创建变量名
df_13
plt.figure(figsize=(6, 4), dpi=1200)
sc = plt.scatter(X_test["Resnet101_ROI"], df_13['df_Resnet101_ROI_INTRAPERI_0-10mm'], 
                 s=10, c=X_test["INTRAPERI_0-10mm"], cmap='coolwarm')
cbar = plt.colorbar(sc, aspect=30, shrink=1)  # 调整颜色条比例和长度
cbar.set_label('INTRAPERI_0-10mm', fontsize=12)  # 设置颜色条标签
cbar.outline.set_visible(False)
plt.axhline(y=0, color='black', linestyle='-.', linewidth=1)  
plt.xlabel('Resnet101_ROI', fontsize=12)
plt.ylabel('SHAP interaction value for\n Resnet101_ROI and INTRAPERI_0-10mm', fontsize=12)  
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("5.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()