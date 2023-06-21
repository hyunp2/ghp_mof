from models import cgcnn

__all__ = ["BACKBONES", "BACKBONE_KWARGS"]

BACKBONES = {
			"cgcnn": cgcnn.CrystalGraphConvNet
			}

#WIP for TorchMDNet Configs!
BACKBONE_KWARGS = {
			"cgcnn": dict(orig_atom_fea_len=92, nbr_fea_len=41,
			 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
			 classification=False, learnable=False, explain=False)
			 }