import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# distribution of SCscore and SAscore
sns.set_theme()
sns.set_style("white")
kwargs = dict(bins=50, stacked=True)

figure_titles = ['Cu paddlewheel-pcu','Zn paddlewheel-pcu','Zn tetramer-pcu']

metric_name = ['SAscore','SCscore']

index=0
fig, ax = plt.subplots(2,3,figsize=(10,5))
for i,metric in enumerate(['sa','sc']):
    for n,sys in enumerate(['CuCu','ZnZn','ZnOZnZnZn']):
        row_num = int(index / 3)
        col_num = index % 3
        for n_atoms in ['n_atoms_5','n_atoms_6','n_atoms_7','n_atoms_8','n_atoms_9','n_atoms_10']:
            if 'n_atoms' in n_atoms:
                # changed to _linker.csv
                data = pd.read_csv(os.path.join('generated_linkers',n_atoms,'sc_sa_score',sys+'_linker.csv'))
                data.drop_duplicates(subset='smiles',keep='first')
                if metric == 'sa':
                   ax[row_num][col_num].hist(data.sa_score,**kwargs,label=n_atoms)
                   ax[row_num][col_num].spines[['left','right', 'top']].set_visible(False)
                else:
                   ax[row_num][col_num].hist(data.sc_score,**kwargs,label=n_atoms)
                   ax[row_num][col_num].spines[['left','right', 'top']].set_visible(False)
        
        ax[row_num][col_num].set_xlabel(metric_name[i])
        ax[row_num][col_num].set_ylabel('Count')
        ax[row_num][col_num].set_title(figure_titles[n])
        index+=1
handles, labels = ax[row_num][col_num].get_legend_handles_labels()

plt.tight_layout()
lgd = fig.legend(handles, labels, loc='lower center', ncol=6,bbox_to_anchor=(0.5, -0.05),framealpha=0,edgecolor='gray')
plt.savefig(f'publication_figures/generated_linkers/dist_sa_sc.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
print(f'Plotted distribution of SAscore and SCscore')