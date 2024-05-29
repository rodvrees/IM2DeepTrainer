import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import logging
import wandb

from utils import BASEMODELCONFIG, MeanMAESorted, LowestMAESorted
# from pytorchsummary import summary

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

class Conv1dActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, initializer, negative_slope, saturation):
        super(Conv1dActivation, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.initializer = initializer
        self.activation = LRelu_with_saturation(negative_slope=negative_slope, saturation=saturation)

        initializer(self.conv.weight, 0.0, 0.05)

    def forward(self, x):
        return self.activation(self.conv(x))

class DenseActivation(nn.Module):
    def __init__(self, in_features, out_features, initializer, negative_slope, saturation):
        super(DenseActivation, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.initializer = initializer
        self.activation = LRelu_with_saturation(negative_slope=negative_slope, saturation=saturation)

        initializer(self.linear.weight, 0.0, 0.05)

    def forward(self, x):
        return self.activation(self.linear(x))

class SelfAttention(nn.Module):
    def __init__(self, feature_dim, heads=1):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.heads = heads
        # self.padded_dim = self.feature_dim + (self.feature_dim % self.heads)
        self.query_dim = self.feature_dim // self.heads
        self.extra_dim = self.feature_dim % self.heads

        self.query = nn.Linear(self.feature_dim, (self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0)) * self.heads)
        self.key = nn.Linear(self.feature_dim, (self.query_dim + (self.extra_dim if self.extra_dim >0 else 0)) * self.heads)
        self.value = nn.Linear(self.feature_dim, (self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0)) * self.heads)

        self.fc_out = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, x):

        batch_size, seq_len, feature_dim = x.size()
        queries = self.query(x).view(batch_size, seq_len, self.heads, self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0))
        keys = self.key(x).view(batch_size, seq_len, self.heads, self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0))
        values = self.value(x).view(batch_size, seq_len, self.heads, self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0))

        attention_scores = torch.einsum('bqhd,bkhd->bhqk', [queries, keys]) / (self.query_dim ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)

        out = torch.einsum('bhqk,bkhd->bqhd', [attention_scores, values])

        out = out.view(batch_size, seq_len, self.heads * (self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0)))
        out = out[:, :, :self.feature_dim]
        out = self.fc_out(out)
        return out

class Branch(nn.Module):
    def __init__(self, input_size, output_size, add_layer=1, dropout_rate=0.0):
        super(Branch, self).__init__()
        self.add_layer = add_layer
        if self.add_layer:
            self.fc1 = nn.Linear(input_size, output_size)
            # self.dropout = nn.Dropout(dropout_rate)
            self.fcoutput = nn.Linear(output_size, 1)
        else:
            self.fcoutput = nn.Linear(input_size, 1)

    def forward(self, x):
        if self.add_layer == 1:
            x = F.relu(self.fc1(x))
            # x = self.dropout(x)
        x = self.fcoutput(x)

        return x

class IM2Deep(L.LightningModule):
    def __init__(self, config, criterion):
        super(IM2Deep, self).__init__()
        self.config = config
        self.criterion = criterion
        self.mae = nn.L1Loss()

        initi = self.configure_init()

        self.ConvAtomComp = nn.ModuleList()
        self.ConvAtomComp.append(
            Conv1dActivation(
                6,
                self.config["AtomComp_out_channels_start"],
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(
            Conv1dActivation(
                self.config["AtomComp_out_channels_start"],
                self.config["AtomComp_out_channels_start"],
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(
            nn.MaxPool1d(
                self.config["AtomComp_MaxPool_kernel_size"],
                self.config["AtomComp_MaxPool_kernel_size"],
            )
        )
        self.ConvAtomComp.append(
            Conv1dActivation(
                self.config["AtomComp_out_channels_start"],
                self.config["AtomComp_out_channels_start"] // 2,
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(
            Conv1dActivation(
                self.config["AtomComp_out_channels_start"] // 2,
                self.config["AtomComp_out_channels_start"] // 2,
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(
            nn.MaxPool1d(
                self.config["AtomComp_MaxPool_kernel_size"],
                self.config["AtomComp_MaxPool_kernel_size"],
            )
        )
        self.ConvAtomComp.append(
            Conv1dActivation(
                self.config["AtomComp_out_channels_start"] // 2,
                self.config["AtomComp_out_channels_start"] // 4,
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(
            Conv1dActivation(
                self.config["AtomComp_out_channels_start"] // 4,
                self.config["AtomComp_out_channels_start"] // 4,
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(nn.Flatten())

        ConvAtomCompSize = (60 // (2 * self.config["AtomComp_MaxPool_kernel_size"])) * (
            self.config["AtomComp_out_channels_start"] // 4
        )

        self.ConvDiatomComp = nn.ModuleList()
        self.ConvDiatomComp.append(
            Conv1dActivation(
                6,
                self.config["DiatomComp_out_channels_start"],
                self.config["DiatomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvDiatomComp.append(
            Conv1dActivation(
                self.config["DiatomComp_out_channels_start"],
                self.config["DiatomComp_out_channels_start"],
                self.config["DiatomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvDiatomComp.append(
            nn.MaxPool1d(
                self.config["DiatomComp_MaxPool_kernel_size"],
                self.config["DiatomComp_MaxPool_kernel_size"],
            )
        )
        self.ConvDiatomComp.append(
            Conv1dActivation(
                self.config["DiatomComp_out_channels_start"],
                self.config["DiatomComp_out_channels_start"] // 2,
                self.config["DiatomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvDiatomComp.append(
            Conv1dActivation(
                self.config["DiatomComp_out_channels_start"] // 2,
                self.config["DiatomComp_out_channels_start"] // 2,
                self.config["DiatomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvDiatomComp.append(nn.Flatten())

        ConvDiAtomCompSize = (30 // self.config["DiatomComp_MaxPool_kernel_size"]) * (
            self.config["DiatomComp_out_channels_start"] // 2
        )

        self.ConvGlobal = nn.ModuleList()
        self.ConvGlobal.append(
            DenseActivation(
                60,
                self.config["Global_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvGlobal.append(
            DenseActivation(
                self.config["Global_units"],
                self.config["Global_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvGlobal.append(
            DenseActivation(
                self.config["Global_units"],
                self.config["Global_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )

        ConvGlobal_output_size = self.config["Global_units"]

        self.OneHot = nn.ModuleList()
        self.OneHot.append(
            Conv1dActivation(
                20,
                self.config["OneHot_out_channels"],
                self.config["One_hot_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.OneHot.append(
            Conv1dActivation(
                self.config["OneHot_out_channels"],
                self.config["OneHot_out_channels"],
                self.config["One_hot_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.OneHot.append(
            nn.MaxPool1d(
                self.config["OneHot_MaxPool_kernel_size"],
                self.config["OneHot_MaxPool_kernel_size"],
            )
        )
        self.OneHot.append(nn.Flatten())

        conv_output_size_OneHot = (60 // self.config["OneHot_MaxPool_kernel_size"]) * self.config[
            "OneHot_out_channels"
        ]

        if config["add_X_mol"]:
            self.MolDesc = nn.ModuleList()
            self.MolDesc.append(
                Conv1dActivation(
                    13,
                    self.config["Mol_out_channels_start"],
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(
                Conv1dActivation(
                    self.config["Mol_out_channels_start"],
                    self.config["Mol_out_channels_start"],
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(
                nn.MaxPool1d(
                    self.config["Mol_MaxPool_kernel_size"],
                    self.config["Mol_MaxPool_kernel_size"],
                )
            )
            self.MolDesc.append(
                Conv1dActivation(
                    self.config["Mol_out_channels_start"],
                    self.config["Mol_out_channels_start"] // 2,
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(
                Conv1dActivation(
                    self.config["Mol_out_channels_start"] // 2,
                    self.config["Mol_out_channels_start"] // 2,
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(
                nn.MaxPool1d(
                    self.config["Mol_MaxPool_kernel_size"],
                    self.config["Mol_MaxPool_kernel_size"],
                )
            )
            self.MolDesc.append(
                Conv1dActivation(
                    self.config["Mol_out_channels_start"] // 2,
                    self.config["Mol_out_channels_start"] // 4,
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(
                Conv1dActivation(
                    self.config["Mol_out_channels_start"] // 4,
                    self.config["Mol_out_channels_start"] // 4,
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(nn.Flatten())

            ConvMolDescSize = (60 // (2 * self.config["Mol_MaxPool_kernel_size"])) * (
                self.config["Mol_out_channels_start"] // 4
            )

        total_input_size = (
            ConvAtomCompSize
            + ConvDiAtomCompSize
            + ConvGlobal_output_size
            + conv_output_size_OneHot
        )

        if config["add_X_mol"]:
            total_input_size += ConvMolDescSize

        self.total_input_size = total_input_size

        self.Concat = nn.ModuleList()
        self.Concat.append(
            DenseActivation(
                self.total_input_size,
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )

        self.Concat.append(nn.Linear(self.config["Concat_units"], 1))

    def regularized_loss(self, y_hat, y):
        standard_loss = self.criterion(y_hat, y)
        l1_norm = sum(torch.norm(p, 1) for p in self.parameters())
        return standard_loss + self.config["L1_alpha"] * l1_norm

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot, mol_desc=None):

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

        if self.config["add_X_mol"]:
            for layer in self.MolDesc:
                mol_desc = layer(mol_desc)

        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)

        if self.config["add_X_mol"]:
            concatenated = torch.cat((concatenated, mol_desc), 1)

        for layer in self.Concat:
            concatenated = layer(concatenated)

        output = concatenated
        return output

    def training_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)

        loss = self.regularized_loss(y_hat, y)

        self.log("Train loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Train MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        loss = self.criterion(y_hat, y)

        self.log("Validation loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Validation MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        loss = self.criterion(y_hat, y)

        self.log("Test loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Test MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    def configure_init(self):
        if (not self.config['init']) or (self.config['init'] == 'normal'):
            return nn.init.normal_
        if self.config['init'] == 'xavier':
            return nn.init.xavier_normal_
        if self.config['init'] == 'kaiming':
            return nn.init.kaiming_normal_

class IM2DeepLSTM(L.LightningModule):
    def __init__(self, config, criterion):
        super(IM2DeepLSTM, self).__init__()
        self.config = config
        self.criterion = criterion
        self.mae = nn.L1Loss()

        initi = self.configure_init()

        self.LSTMAtomComp = nn.LSTM(
            6,
            1024,
            num_layers=3,
            batch_first=True,
            bidirectional=True)

        self.LSTMDiatomComp = nn.LSTM(
            6,
            512,
            num_layers=3,
            batch_first=True,
            bidirectional=True)

        self.Global = nn.ModuleList()
        self.Global.append(
            DenseActivation(
                60,
                self.config["Global_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Global.append(
            DenseActivation(
                self.config["Global_units"],
                self.config["Global_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Global.append(
            DenseActivation(
                self.config["Global_units"],
                self.config["Global_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )

        self.OneHot = nn.LSTM(
            20,
            80,
            num_layers=3,
            batch_first=True,
            bidirectional=True)

        if config["add_X_mol"]:
            self.MolDesc = nn.LSTM(
                13,
                256,
                num_layers=2,
                batch_first=True,
                bidirectional=True)

        total_input_size = 1024*2 + 512*2 + 256 + 80*2
        if config["add_X_mol"]:
            total_input_size += 256*2

        self.total_input_size = total_input_size

        self.Concat = nn.ModuleList()
        self.Concat.append(
            DenseActivation(
                self.total_input_size,
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )

        self.Concat.append(nn.Linear(self.config["Concat_units"], 1))

    def regularized_loss(self, y_hat, y):
        standard_loss = self.criterion(y_hat, y)
        l1_norm = sum(torch.norm(p, 1) for p in self.parameters())
        return standard_loss + self.config["L1_alpha"] * l1_norm

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot, mol_desc=None):
        if self.config["add_X_mol"]:
            mol_desc = mol_desc.permute(0, 2, 1)

        atom_comp, _ = self.LSTMAtomComp(atom_comp)
        diatom_comp, _ = self.LSTMDiatomComp(diatom_comp)

        for layer in self.Global:
            global_feats = layer(global_feats)

        one_hot, _ = self.OneHot(one_hot)

        if self.config["add_X_mol"]:
            mol_desc, _ = self.MolDesc(mol_desc)

        concatenated = torch.cat((atom_comp[:, -1, :], diatom_comp[:, -1, :], one_hot[:, -1, :], global_feats), 1)

        if self.config["add_X_mol"]:
            concatenated = torch.cat((concatenated, mol_desc[:, -1, :]), 1)

        for layer in self.Concat:
            concatenated = layer(concatenated)

        output = concatenated
        return output

    def training_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)

        loss = self.regularized_loss(y_hat, y)

        self.log("Train loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Train MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        loss = self.criterion(y_hat, y)

        self.log("Validation loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Validation MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        loss = self.criterion(y_hat, y)

        self.log("Test loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Test MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    def configure_init(self):
        if (not self.config['init']) or (self.config['init'] == 'normal'):
            return nn.init.normal_
        if self.config['init'] == 'xavier':
            return nn.init.xavier_normal_
        if self.config['init'] == 'kaiming':
            return nn.init.kaiming_normal_



class IM2DeepMulti(L.LightningModule):
    def __init__(self, config, criterion):
        super(IM2DeepMulti, self).__init__()
        # TODO: config should be adapted in config file
        self.config = config
        self.criterion = criterion
        self.l1_alpha = config['L1_alpha']

        # Load the IM2Deep model
        try:
            self.backbone(IM2Deep(config['backbone_config']))
        except KeyError:
            self.backbone = IM2Deep(BASEMODELCONFIG)

        try:
            self.backbone.load_state_dict(torch.load(config['backbone_SD_path']))
        except KeyError:
            self.backbone.load_state_dict(torch.load('/home/robbe/IM2DeepMulti/BestParams_final_model_state_dict.pth'))

        self.ConvAtomComp = self.backbone.ConvAtomComp
        self.ConvDiatomComp = self.backbone.ConvDiatomComp
        self.ConvGlobal = self.backbone.ConvGlobal
        self.OneHot = self.backbone.OneHot

        self.concat = list(self.backbone.Concat.children())[:-1]

        if self.config['Use_attention_concat'] == 1:
            # TODO: don't hardcode the input size, write a function to get the input size
            self.SelfAttentionConcat = SelfAttention(1841, config['Concatheads'])
        if self.config['Use_attention_output'] == 1:
            self.SelfAttentionOutput = SelfAttention(94, config['Outputheads'])

        # TODO: idem
        self.branches = nn.ModuleList([Branch(94, config['BranchSize'], add_layer=config['Add_branch_layer']),
                                       Branch(94, config['BranchSize'], add_layer=config['Add_branch_layer'])])

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot, mol_desc=None):
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

        # TODO: mol_desc for multioutput

        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)

        if self.config['Use_attention_concat'] == 1:
            concatenated = self.SelfAttentionConcat(concatenated.unsqueeze(1)).squeeze(1)

        for layer in self.concat:
            concatenated = layer(concatenated)

        if self.config['Use_attention_output'] == 1:
            concatenated = self.SelfAttentionOutput(concatenated.unsqueeze(1)).squeeze(1)

        y_hat1 = self.branches[0](concatenated)
        y_hat2 = self.branches[1](concatenated)

        return y_hat1, y_hat2

    def training_step(self, batch, batch_idx):
        atom_comp, diatom_comp, global_feats, one_hot, y = batch
        y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)

        y1, y2 = y[:, 0], y[:, 1]

        loss = self.criterion(y1, y2, y_hat1, y_hat2)

        l1_norm = sum(p.abs().sum() for p in self.parameters())
        total_loss = loss + self.l1_alpha * l1_norm

        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log('Train Loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        atom_comp, diatom_comp, global_feats, one_hot, y = batch
        y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)

        y1, y2 = y[:, 0], y[:, 1]

        loss = self.criterion(y1, y2, y_hat1, y_hat2)

        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log('Val Loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Val Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Val Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        atom_comp, diatom_comp, global_feats, one_hot, y = batch
        y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)

        y1, y2 = y[:, 0], y[:, 1]

        loss = self.criterion(y1, y2, y_hat1, y_hat2)
        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log('Test Loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Test Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Test Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch):
        atom_comp, diatom_comp, global_feats, one_hot, y = batch
        y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)
        return torch.hstack([y_hat1, y_hat2])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer
