import math
import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.traffic_speed_prediction.STGCN import STGCN



class PPC(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.drop_prob = config.get('dropout', 0)
        self.device = config.get('device', torch.device('cpu'))
        self.embed_dim = config.get('embed_dim')
        self.num_layers = config.get('num_layers')
        self.num_heads = config.get('num_heads')
        self.embedding_layer = torch.nn.Linear(self.input_window,self.embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dropout=self.drop_prob)
        self.backbone_model = torch.nn.TransformerEncoder(self.encoder_layer, self.num_layers)
        self.output_layer = torch.nn.Linear(self.embed_dim,self.output_window)

    def forward(self, batch):
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        x = x.permute(2, 0, 1, 3)  # (num_nodes, batch_size, input_length, feature_dim)
        x = torch.squeeze(x,axis=3) # (num_nodes, batch_size, input_length)
        x_c = torch.clone(x) # (num_nodes, batch_size, input_length)
        x_c = x_c.permute(1,2,0)
        x_c_t = x_c.permute(0,2,1)
        x_s = torch.bmm(x_c,x_c_t)
        x_s = x_s.permute(1,0,2)
        x_s = self._scaler.transform(x_s)
        x_input = torch.cat((x_s, x), dim=0)
        x_embed = self.embedding_layer(x_input)
        x_embed = self.activation(x_embed)
        x_embed = self.dropout(x_embed)
        x_output = self.backbone_model(x_embed)
        final_output = self.output_layer(x_output)
        token,final_output = torch.split(final_output,[self.input_window,self.num_nodes],dim=0)
        final_output = torch.unsqueeze(final_output,-1) 
        outputs = final_output.permute(1, 2, 0, 3)  # (batch_size, output_length(1), num_nodes, output_dim)
        return outputs

    def calculate_loss(self, batch):
        y_true = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
        y_predicted = self.predict(batch)  # (batch_size, output_length, num_nodes, output_dim)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        y_predicted = self.forward(batch) 
        return y_predicted
