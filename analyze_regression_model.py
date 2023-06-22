import warnings
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# predictions of pre-trained model on the test set
sns.set_theme(style='white', rc={'axes.linewidth': 1, 'axes.edgecolor':'black'})
plt.rcParams.update({'font.size': 40})

data = pd.read_csv('./model_predictions/predictions_hMOF_test.csv')

r2 = r2_score(data['pred'],data['target'])
mae = mean_absolute_error(data['pred'],data['target'])
rmse = np.sqrt(mean_squared_error(data['pred'],data['target']))

# y = x line
data['y_line'] = data['pred']

plot = sns.relplot(data=data, x='pred', y='y_line',kind='line',linestyle='--',color='k',linewidth=1)
plot.map_dataframe(sns.scatterplot, 'pred', 'target', color='k',s=20)
plt.fill_between([-0.5,2], -0.3, 2, color='yellow', alpha=0.3, interpolate=True)
plt.fill_between([2,6.1], -0.3, 2, color='tab:purple', alpha=0.3, interpolate=True)
plt.fill_between([-0.5,2], 2, 6.1, color='tab:red', alpha=0.3, interpolate=True)
plt.fill_between([2,6.1], 2, 6.1, color='tab:green', alpha=0.3, interpolate=True)
plt.xlabel('Predicted CO$_\mathrm{2}$ capacity (mmol/g)', labelpad=4)
plt.ylabel('True CO$_\mathrm{2}$ capacity (mmol/g)')
plt.axvline(2,linestyle='--',color='k',linewidth=1)
plt.axhline(2,linestyle='--',color='k',linewidth=1)

for ax in plot.axes.flatten(): # Loop directly on the flattened axes 
    for _, spine in ax.spines.items():
        spine.set_visible(True) # You have to first turn them on
        spine.set_color('black')
        spine.set_linewidth(1)

plt.xlim([-0.3,6.1])
plt.ylim([-0.3,6.1])
plt.yticks(np.arange(0,7,1))
plt.title('R$^\mathrm{2}$:'+f'{r2:.2f}   MAE: {mae:.2f} RMSE: {rmse:.2f}',pad=10)
plt.tight_layout()
plt.savefig('publication_figures/regression_model_inference_hMOF/confusion_matrix.pdf',bbox_inches='tight')
print('Plotted confusion matrix')

# predictions of pre-trained model on generated structures

# histogram of standard deviation
df_pred = pd.read_csv('./model_predictions/predictions_generated.csv')
sns.set_theme(style='white')
sns.displot(df_pred['std'],bins=1000,kde=False,height=4,aspect=5.5/4)
plt.axvline(0.2,c='k',linewidth=1,linestyle='--')
plt.xlabel('standard deviation (mmol/g)')
plt.xlim([0,0.4])
plt.savefig('publication_figures/regression_model_inference_generated/dist_std.pdf',bbox_inches='tight')

# ecdf - capacity
df_hMOF = pd.read_csv('./data/hMOF_CO2_info_node_linker.csv')

nodes = ['[Cu][Cu]','[Zn][Zn]','[Zn][O]([Zn])([Zn])[Zn]']
node_names = ['CuCu','ZnZn','ZnOZnZnZn'] 
node_names_full = ['Cu paddlewheel-pcu','Zn paddlewheel-pcu','Zn tetramer-pcu']

fig, ax = plt.subplots(2,2,figsize=(8,8))
for i in range(4):
    row = int(i / 2)
    col = i % 2

    df_hMOF_select = df_hMOF[df_hMOF.MOFid.str.contains(f'cat{i}')]
    df_hMOF_select_CuCu = df_hMOF_select[df_hMOF_select.node == '[Cu][Cu]']
    df_hMOF_select_ZnZn = df_hMOF_select[df_hMOF_select.node == '[Zn][Zn]']
    df_hMOF_select_ZnOZnZnZn = df_hMOF_select[df_hMOF_select.node == '[Zn][O]([Zn])([Zn])[Zn]']

    if i == 0:
        df_pred_select = df_pred[(~df_pred.mof_name.str.contains('cat1'))&(~df_pred.mof_name.str.contains('cat2'))&(~df_pred.mof_name.str.contains('cat3'))]
    else:
        df_pred_select = df_pred[df_pred.mof_name.str.contains(f'cat{i}')]
    df_pred_select_CuCu = df_pred_select[df_pred_select.mof_name.str.contains('-CuCu-')]
    df_pred_select_ZnZn = df_pred_select[df_pred_select.mof_name.str.contains('-ZnZn-')]
    df_pred_select_ZnOZnZnZn = df_pred_select[df_pred_select.mof_name.str.contains('-ZnOZnZnZn-')]

    df_hMOF_select_CuCu['wc'] = df_hMOF_select_CuCu['CO2_capacity_01']
    df_hMOF_select_CuCu['node type'] = 'hMOF_Cu_paddlewheel_pcu'
    df_hMOF_select_CuCu = df_hMOF_select_CuCu.loc[:,['wc','node type']]

    df_hMOF_select_ZnZn['wc'] = df_hMOF_select_ZnZn['CO2_capacity_01']
    df_hMOF_select_ZnZn['node type'] = 'hMOF_Zn_paddlewheel_pcu'
    df_hMOF_select_ZnZn = df_hMOF_select_ZnZn.loc[:,['wc','node type']]

    df_hMOF_select_ZnOZnZnZn['wc'] = df_hMOF_select_ZnOZnZnZn['CO2_capacity_01']
    df_hMOF_select_ZnOZnZnZn['node type'] = 'hMOF_Zn_tetramer_pcu'
    df_hMOF_select_ZnOZnZnZn = df_hMOF_select_ZnOZnZnZn.loc[:,['wc','node type']]
    
    df_pred_select_CuCu['wc'] = df_pred_select_CuCu['avg']
    df_pred_select_CuCu['node type'] = 'gen_Cu_paddlewheel_pcu'
    df_pred_select_CuCu = df_pred_select_CuCu.loc[:,['wc','node type']]

    df_pred_select_ZnZn['wc'] = df_pred_select_ZnZn['avg']
    df_pred_select_ZnZn['node type'] = 'gen_Zn_paddlewheel_pcu'
    df_pred_select_ZnZn = df_pred_select_ZnZn.loc[:,['wc','node type']]

    df_pred_select_ZnOZnZnZn['wc'] = df_pred_select_ZnOZnZnZn['avg']
    df_pred_select_ZnOZnZnZn['node type'] = 'gen_Zn_tetramer_pcu'
    df_pred_select_ZnOZnZnZn = df_pred_select_ZnOZnZnZn.loc[:,['wc','node type']]
    
    sns.ecdfplot(ax=ax[row, col],data=df_hMOF_select_CuCu, x="wc", hue="node type",linewidth=2,label='hMOF_Cu_paddlewheel_pcu',legend=False,palette=['tab:blue'],linestyle='--')
    sns.ecdfplot(ax=ax[row, col],data=df_hMOF_select_ZnZn, x="wc", hue="node type",linewidth=2,label='hMOF_Zn_paddlewheel_pcu',legend=False,palette=['tab:green'],linestyle='--')
    sns.ecdfplot(ax=ax[row, col],data=df_hMOF_select_ZnOZnZnZn, x="wc", hue="node type",linewidth=2,label='hMOF_Zn_tetramer_pcu',legend=False,palette=['tab:orange'],linestyle='--')
    sns.ecdfplot(ax=ax[row, col],data=df_pred_select_CuCu, x="wc", hue="node type",linewidth=2,label='generated_Cu_paddlewheel_pcu',legend=False,palette=['tab:blue'])
    sns.ecdfplot(ax=ax[row, col],data=df_pred_select_ZnZn, x="wc", hue="node type",linewidth=2,label='generated_Zn_paddlewheel_pcu',legend=False,palette=['tab:green'])
    sns.ecdfplot(ax=ax[row, col],data=df_pred_select_ZnOZnZnZn, x="wc", hue="node type",linewidth=2,label='generated_Zn_tetramer_pcu',legend=False,palette=['tab:orange'])
    ax[row, col].axvline(2,linestyle='--',c='k',linewidth=1)
    ax[row, col].xaxis.set_tick_params(labelsize=12)
    ax[row, col].xaxis.set_tick_params(labelsize=12)
    ax[row, col].set_xlabel('CO$_\mathrm{2}$ capacity @ 0.1 bar (mmol/g)', fontsize=12)
    ax[row, col].set_ylabel('Density', fontsize=12)
    ax[row, col].set_title(f'cat{i}',fontsize=16,pad=8)
    ax[row, col].set_xlim([0,4])

handles, labels = ax[row][col].get_legend_handles_labels()

lgd = fig.legend(handles, labels, ncol=2, loc='lower center',bbox_to_anchor=(0.5, -0.1),framealpha=0,edgecolor='gray')

plt.tight_layout()
plt.savefig(f'publication_figures/regression_model_inference_generated/dist_pred.pdf',bbox_inches='tight')
plt.show()