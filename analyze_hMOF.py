import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

df = pd.read_csv('data/hMOF_CO2_info_node_linker.csv')
df_select = df[df.node.isin(['[Cu][Cu]','[Zn][Zn]','[Zn][O]([Zn])([Zn])[Zn]'])]

# ecdf - node
plt.rcParams.update({'font.size': 16})
df_select_CuCu = df_select[df_select.node=='[Cu][Cu]']
df_select_ZnZn = df_select[df_select.node=='[Zn][Zn]']
df_select_ZnOZnZnZn = df_select[df_select.node=='[Zn][O]([Zn])([Zn])[Zn]']

fig,ax = plt.subplots()
sns.ecdfplot(df_select_CuCu,x='CO2_capacity_01',hue='node',linewidth=2,palette=['tab:blue'],label='Cu paddlewheel-pcu')
sns.ecdfplot(df_select_ZnZn,x='CO2_capacity_01',hue='node',linewidth=2,palette=['tab:green'],label='Zn paddlewheel-pcu')
sns.ecdfplot(df_select_ZnOZnZnZn,x='CO2_capacity_01',hue='node',linewidth=2,palette=['tab:orange'],label='Zn tetramer-pcu')
handles, labels = ax.get_legend_handles_labels()
plt.legend(loc='lower right')
plt.xlim([0,4])
plt.xlabel('CO$_\mathrm{2}$ capacity (mmol/g)', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('publication_figures/hMOF/ecdf_node.pdf',bbox_inches='tight')
print('Plotted ecdf_node')


# ecdf - catenation
plt.rcParams.update({'font.size': 16})
cat = []
for i in list(df_select.MOFid):
    try:
        cat.append('cat'+str(int(i.split('cat')[1].split(';')[0])))
    except:
        cat.append(np.NaN)
# append new cat column to dataframe
df_select['cat'] = cat
# remove invalid MOFids
df_select.dropna(inplace=True)

df_cat0 = df_select[df_select.cat=='cat0']
df_cat1 = df_select[df_select.cat=='cat1']
df_cat2 = df_select[df_select.cat=='cat2']
df_cat3 = df_select[df_select.cat=='cat3']
df_all = pd.concat([df_cat0,df_cat1,df_cat2,df_cat3],axis=0)

fig,ax = plt.subplots(figsize=(6,5))

g = sns.ecdfplot(data=df_all, x="CO2_capacity_01", hue="cat",linewidth=2,palette=['k','#8C564B','#9467BD','#FF7F0E'],ax=ax)
g.legend_.set_title(None)
plt.xlabel('CO$_\mathrm{2}$ capacity (mmol/g)')
plt.ylabel('Density')
plt.axvline(2,linestyle='--',c='k',linewidth=1)
plt.xlim([0,4])
plt.tight_layout()
plt.savefig('publication_figures/hMOF/ecdf_cat.pdf',bbox_inches='tight')
print('Plotted ecdf_cat')

# CO2 capacity pairplot
plt.rcParams.update({'font.size': 25})
node_topology = []
for n in df_select.node:
    if n == '[Cu][Cu]':
        node_topology.append('Cu paddlewheel-pcu')
    if n == '[Zn][Zn]':
        node_topology.append('Zn paddlewheel-pcu')
    if n == '[Zn][O]([Zn])([Zn])[Zn]':
        node_topology.append('Zn tetramer-pcu')

df_select['node-topology'] = node_topology

# rename columns for plotting
df_select.columns = ['MOF','MOFid','0.01bar','0.05bar','0.1bar','0.5bar','2.5bar','metal_node','organic_linkers','cat','node-topology']

df_select_CuCu = df_select[df_select.metal_node == '[Cu][Cu]']
df_select_ZnZn = df_select[df_select.metal_node == '[Zn][Zn]']
df_select_ZnOZnZnZn = df_select[df_select.metal_node == '[Zn][O]([Zn])([Zn])[Zn]']
df_select_new = pd.concat([df_select_CuCu,df_select_ZnZn,df_select_ZnOZnZnZn],axis=0)

os.makedirs('publication_figures/hMOF/',exist_ok=True)
ax = sns.pairplot(df_select_new,hue='node-topology',corner=True,palette=['tab:blue','tab:green','tab:orange'],kind='scatter',plot_kws={'alpha':0.5})
plt.tight_layout()
plt.savefig('publication_figures/hMOF/capacity_pairplot.png', dpi=300, bbox_inches="tight")
print('Plotted capacity_pairplot')