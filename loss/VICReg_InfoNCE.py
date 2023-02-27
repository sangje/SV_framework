from InfoNCE import InfoNCE
from VICReg import VICReg
import torch.nn as nn

class VICReg_InfoNCE(nn.Module):

    def __init__(
        self,
        VIC_weight=1,
        Info_weight=1,
        inv_weight=1.0,
        var_weight=1.0,
        cov_weight=0.04
    ):
        super().__init__()

        self.VIC_weight = VIC_weight
        self.Info_weight = Info_weight
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self,data):
        
        VIC=VICReg(inv_weight=self.inv_weight,
                   var_weight=self.var_weight,
                   cov_weight=self.cov_weight)
        Info=InfoNCE()

        VIC_loss = VIC(data)
        Info_loss, acc = Info(data)

        loss=self.VIC_weight*VIC_loss + self.Info_weight*Info_loss
        return loss, acc