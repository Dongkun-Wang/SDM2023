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
# from libcity.executor.traffic_state_executor import TrafficStateExecutor





class PPD(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.distill = config.get('distill')
        self.drop_prob = config.get('dropout', 0)
        self.device = config.get('device', torch.device('cpu'))
        self.spatial_embed_dim = config.get('spatial_embed_dim')
        self.temporal_embed_dim = config.get('temporal_embed_dim')
        self.spatial_num_layers = config.get('spatial_num_layers')
        self.temporal_num_layers = config.get('temporal_num_layers')
        self.num_heads = config.get('num_heads')
        self.epoch = config.get('epoch')
        self.max_epoch = config.get('max_epoch')

        self.spatial_embedding_layer = torch.nn.Linear(self.input_window, self.spatial_embed_dim)
        self.temporal_embedding_layer = torch.nn.Linear(self.num_nodes, self.temporal_embed_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        
        self.spatial_encoder_layer = nn.TransformerEncoderLayer(d_model=self.spatial_embed_dim, nhead=self.num_heads, dropout=self.drop_prob)
        self.spatial_model = torch.nn.TransformerEncoder(self.spatial_encoder_layer, self.spatial_num_layers)
        
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=self.temporal_embed_dim, nhead=self.num_heads, dropout=self.drop_prob)
        self.temporal_model = torch.nn.TransformerEncoder(self.temporal_encoder_layer, self.temporal_num_layers)
        
        self.spatial_output_layer = torch.nn.Linear(self.spatial_embed_dim, self.output_window)
        self.temporal_output_layer = torch.nn.Linear(self.temporal_embed_dim, self.num_nodes)

        self.final_output_layer = torch.nn.Linear(2 * self.output_window, self.output_window)

    def forward(self, batch):
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        x = torch.squeeze(x,axis=3)  # (batch_size, input_length, num_nodes)
        # spatial transformer
        x_spatial = x.permute(2, 0, 1)  # (num_nodes, batch_size, input_length)
        x_spatial_embed = self.spatial_embedding_layer(x_spatial)
        #x_spatial_embed = self.activation(x_spatial_embed)
        #x_spatial_embed = self.dropout(x_spatial_embed)
        x_spatial_output = self.spatial_model(x_spatial_embed)
        spatial_final_output = self.spatial_output_layer(x_spatial_output)

        # temporal transformer
        x_temporal = x.permute(1, 0, 2)  # (num_nodes, batch_size, input_length)
        x_temporal_embed = self.temporal_embedding_layer(x_temporal)
        #x_temporal_embed = self.activation(x_temporal_embed)
        x#_temporal_embed = self.dropout(x_temporal_embed)
        x_temporal_output = self.temporal_model(x_temporal_embed)
        temporal_final_output = self.temporal_output_layer(x_temporal_output)
        temporal_final_output = temporal_final_output.permute(2,1,0)

        final_output_c = torch.cat((spatial_final_output,temporal_final_output), dim=2)
        final_output = self.final_output_layer(final_output_c)
        final_output = torch.unsqueeze(final_output, dim=-1)
        outputs = final_output.permute(1, 2, 0, 3)  # (batch_size, output_length(1), num_nodes, output_dim)
        return outputs

    def calculate_loss(self, batch, teacher_output=None):
        y_true = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
        if teacher_output!= None:
            if self.distill == "hard":
                y_true = 0.95*teacher_output +0.05*y_true       
            elif self.distill == "soft":
                y_true = 0.05*teacher_output +0.95*y_true
            elif self.distill == "avg":
                y_true = 0.5*teacher_output +0.5*y_true
            elif self.distill == "moment":
                y_true = (1-(self.epoch/self.max_epoch))*teacher_output +(self.epoch/self.max_epoch)*y_true
        y_predicted = self.predict(batch)  # (batch_size, output_length, num_nodes, output_dim)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        y_predicted = self.forward(batch) 
        return y_predicted
