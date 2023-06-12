from data_utils import *
from dist_utils import *
from loss_utils import *
from train_utils import *

if __name__ == "__main__":
    config = dict(orig_atom_fea_len=92, nbr_fea_len=41,
         atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
         classification=False)
    m = CrystalGraphConvNet(**config)
    
    
    
