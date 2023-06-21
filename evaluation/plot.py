import matplotlib.pyplot as plt
import numpy as np
import math

##### Plotting #####

def plot_hist(results, prop_name, labels=None, alpha_val=0.5):
    if labels is None:
        labels = [str(i) for i in range(len(results))]
    results_all = []
    for res in results:
        results_all.extend(res)
        
    min_value = np.amin(results_all)
    max_value = np.amax(results_all)
    num_bins = 20.0
    binwidth = (max_value - min_value) / num_bins
    
    if prop_name in ['heavy_atoms', 'num_C', 'num_N', 'num_O', 'num_F', 'hba', 'hbd', 'ring_ct', 'avg_ring_size', 'rot_bnds', 'stereo_cnts', "Linker lengths"]:
        min_value = math.floor(min_value)
        max_value = math.ceil(max_value)
        diff = max_value - min_value
        binwidth = max(1, int(diff/num_bins))
        
    if prop_name in ["Mol_similarity", "QED"]:
        min_value = 0.0
        max_value = 1.01
        diff = max_value - min_value
        binwidth = diff/num_bins
        
    if prop_name in ["SC_RDKit"]:
        max_value = 1.01
        diff = max_value - min_value
        binwidth = diff/num_bins
 
    bins = np.arange(min_value - 2* binwidth, max_value + 2* binwidth, binwidth)
    
    dens_all = []
    for i, res in enumerate(results):
        if not labels:
            dens, _ , _ = plt.hist(np.array(res).flatten(), bins=bins, density=True, alpha=alpha_val)
        elif labels:
            dens, _ , _ = plt.hist(np.array(res).flatten(), bins=bins, density=True, alpha=alpha_val,label=labels[i])
        #dens, _ , _ = plt.hist(res, bins=bins, density=True, alpha=alpha_val, label=labels[i])
        dens_all.extend(dens)

    plt.xlabel(prop_name)
    plt.ylabel('Proportion')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1*0.8,x2/0.8,y1,1.1*max(dens_all)))
    plt.title('Distribution of ' + prop_name)
    plt.legend()
    plt.grid(True)

    plt.show()