import operator, functools
import torch
from typing import *

def get_loss_func(opt, pred: List[torch.Tensor], targetE: torch.Tensor, targetF: torch.Tensor, coeffs=[0.1,0.9]):
#     with_force = opt.with_force #DEPRECATED: False is default
    with torch.cuda.amp.autocast(enabled=opt.amp):
        E, F = pred
        E = E.view(-1)
        F = F.to(E) #Ensure same dtype
        F = F.view(-1,3)
        targetE = targetE.to(E).view(-1) #Ensure same dtype
        targetF = targetF.to(E).view(-1,3) #Ensure same dtype

        Etruth = targetE
        Ftruth = None if targetF.count_nonzero().item() == 0 else targetF
    loss_e = torch.nn.functional.mse_loss(E, Etruth, reduction="mean")
#     loss_f = torch.mean(torch.norm((F - Ftruth), p=2, dim=1))
    if Ftruth != None:
        loss_f = torch.nn.functional.mse_loss(F, Ftruth, reduction="mean")
        losses = [loss_e, loss_f]
    else:
        losses = [loss_e]
        coeffs = [1.0]
        
    assert isinstance(losses, list) and isinstance(coeffs, list), "incompatible formats for losses and coeffs..."
    assert len(losses) == len(coeffs), "must have matching length..."
    
    loss = functools.reduce(operator.add, [l*c for l, c in zip(losses, coeffs)] ) #total loss
    return loss

def get_loss_func_crystal(opt, pred: torch.Tensor, targetE: torch.Tensor):
    with torch.cuda.amp.autocast(enabled=opt.amp):
        E = pred.view(-1,)
        targetE = targetE.to(E).view(-1,) #Ensure same dtype
        Etruth = targetE

    loss_e = torch.nn.functional.mse_loss(E, Etruth, reduction="mean")
    loss = loss_e
    return loss

