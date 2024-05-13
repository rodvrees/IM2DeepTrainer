import torch
import torch.nn as nn
import lightning as L
import logging
import wandb

logger = logging.getLogger(__name__)

class LogLowestMAE(L.Callback):
    def __init__(self):
        super(LogLowestMAE, self).__init__()
        self.bestMAE = float("inf")

    def on_validation_end(self, trainer, pl_module):
        currentMAE = trainer.callback_metrics["Validation MAE"]
        if currentMAE < self.bestMAE:
            self.bestMAE = currentMAE
        wandb.log({"Best Val MAE": self.bestMAE})


class LRelu_with_saturation(nn.Module):
    def __init__(self, negative_slope, saturation):
        super(LRelu_with_saturation, self).__init__()
        self.negative_slope = negative_slope
        self.saturation = saturation
        self.leaky_relu = nn.LeakyReLU(self.negative_slope)

    def forward(self, x):
        activated = self.leaky_relu(x)
        return torch.clamp(activated, max=self.saturation)

class IM2Deep(L.LightningModule):
    def __init__(self, config, criterion):
        super(IM2Deep, self).__init__()
        self.config = config
        self.criterion = criterion
        self.mae = nn.L1Loss()

        self.ConvAtomComp = nn.ModuleList()
        self.ConvAtomComp.append(nn.Conv1d(6, self.config['AtomComp_out_channels_start'], self.config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(self.config['AtomComp_out_channels_start'], self.config['AtomComp_out_channels_start'], self.config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.MaxPool1d(self.config['AtomComp_MaxPool_kernel_size'], self.config['AtomComp_MaxPool_kernel_size']))
        self.ConvAtomComp.append(nn.Conv1d(self.config['AtomComp_out_channels_start'], self.config['AtomComp_out_channels_start']//2, self.config['AtomComp_kernel_size'], padding='same')) #Input is probably 256 now?
        self.ConvAtomComp.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(self.config['AtomComp_out_channels_start']//2, self.config['AtomComp_out_channels_start']//2, self.config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.MaxPool1d(self.config['AtomComp_MaxPool_kernel_size'], self.config['AtomComp_MaxPool_kernel_size']))
        self.ConvAtomComp.append(nn.Conv1d(self.config['AtomComp_out_channels_start']//2, self.config['AtomComp_out_channels_start']//4, self.config['AtomComp_kernel_size'], padding='same')) #Input is probably 128 now?
        self.ConvAtomComp.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(self.config['AtomComp_out_channels_start']//4, self.config['AtomComp_out_channels_start']//4, self.config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Flatten())
        ConvAtomCompSize = (60 // (2 * self.config['AtomComp_MaxPool_kernel_size'])) * (self.config['AtomComp_out_channels_start']//4)

        self.ConvDiatomComp = nn.ModuleList()
        self.ConvDiatomComp.append(nn.Conv1d(6, self.config['DiatomComp_out_channels_start'], self.config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.Conv1d(self.config['DiatomComp_out_channels_start'], self.config['DiatomComp_out_channels_start'], self.config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.MaxPool1d(self.config['DiatomComp_MaxPool_kernel_size'], self.config['DiatomComp_MaxPool_kernel_size']))
        self.ConvDiatomComp.append(nn.Conv1d(self.config['DiatomComp_out_channels_start'], self.config['DiatomComp_out_channels_start']//2, self.config['DiatomComp_kernel_size'], padding='same')) #Input is probably 64 now?
        self.ConvDiatomComp.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.Conv1d(self.config['DiatomComp_out_channels_start']//2, self.config['DiatomComp_out_channels_start']//2, self.config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.Flatten())

        # Calculate the output size of the DiatomComp layers
        ConvDiAtomCompSize = (30 // self.config['DiatomComp_MaxPool_kernel_size']) * (self.config['DiatomComp_out_channels_start']//2)

        self.ConvGlobal = nn.ModuleList()
        self.ConvGlobal.append(nn.Linear(60, self.config['Global_units']))
        self.ConvGlobal.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvGlobal.append(nn.Linear(self.config['Global_units'], self.config['Global_units']))
        self.ConvGlobal.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.ConvGlobal.append(nn.Linear(self.config['Global_units'], self.config['Global_units']))
        self.ConvGlobal.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))

        # Calculate the output size of the Global layers
        ConvGlobal_output_size = self.config['Global_units']

        # One-hot encoding
        self.OneHot = nn.ModuleList()
        self.OneHot.append(nn.Conv1d(20, self.config['OneHot_out_channels'], self.config['One_hot_kernel_size'], padding='same'))
        self.OneHot.append(nn.Tanh())
        self.OneHot.append(nn.Conv1d(self.config['OneHot_out_channels'], self.config['OneHot_out_channels'], self.config['One_hot_kernel_size'], padding='same'))
        self.OneHot.append(nn.Tanh())
        self.OneHot.append(nn.MaxPool1d(self.config['OneHot_MaxPool_kernel_size'], self.config['OneHot_MaxPool_kernel_size']))
        self.OneHot.append(nn.Flatten())

        # Calculate the output size of the OneHot layers
        conv_output_size_OneHot = ((60 // self.config['OneHot_MaxPool_kernel_size']) * self.config['OneHot_out_channels'])
        print(conv_output_size_OneHot)

        # Calculate the total input size for the Concat layer
        total_input_size = ConvAtomCompSize + ConvDiAtomCompSize + ConvGlobal_output_size + conv_output_size_OneHot
        print(total_input_size)

        self.total_input_size = total_input_size

        # Concatenate
        self.Concat = nn.ModuleList()
        self.Concat.append(nn.Linear(total_input_size, self.config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.Concat.append(nn.Linear(self.config['Concat_units'], self.config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.Concat.append(nn.Linear(self.config['Concat_units'], self.config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.Concat.append(nn.Linear(self.config['Concat_units'], self.config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))
        self.Concat.append(nn.Linear(self.config['Concat_units'], self.config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(self.config['LRelu_negative_slope'], self.config['LRelu_saturation']))

        self.Concat.append(nn.Linear(self.config['Concat_units'], 1))

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        for layer in self.ConvAtomComp:
            atom_comp = layer(atom_comp)

        for layer in self.ConvDiatomComp:
            diatom_comp = layer(diatom_comp)
        for layer in self.ConvGlobal:
            global_feats = layer(global_feats)
        for layer in self.OneHot:
            one_hot = layer(one_hot)

        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)

        for layer in self.Concat:
            concatenated = layer(concatenated)

        output = concatenated
        return output

    def training_step(self, batch, batch_idx):
        atom_comp, diatom_comp, global_feats, one_hot, y = batch
        # logger.debug(y)
        y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        # logger.debug(y_hat)
        loss = self.criterion(y_hat, y)



        self.log('Train loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train MAE', self.mae(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        atom_comp, diatom_comp, global_feats, one_hot, y = batch
        y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        loss = self.criterion(y_hat, y)

        self.log('Validation loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Validation MAE', self.mae(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        atom_comp, diatom_comp, global_feats, one_hot, y = batch
        y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        loss = self.criterion(y_hat, y)

        self.log('Test loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Test MAE', self.mae(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        atom_comp, diatom_comp, global_feats, one_hot, y = batch
        y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer
