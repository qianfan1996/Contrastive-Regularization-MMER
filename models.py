import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

class AttentionLSTM(torch.nn.Module):
    """Taken from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py"""
    def __init__(self, cfg):
        """
        LSTM with self-Attention model.
        :param cfg: Linguistic config object
        """
        super(AttentionLSTM, self).__init__()
        self.batch_size = cfg.batch_size
        self.output_size = cfg.num_classes
        self.hidden_size = cfg.hidden_dim
        self.embedding_length = cfg.emb_dim
        self.bidirectional = cfg.bidirectional

        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.dropout2 = torch.nn.Dropout(cfg.dropout2)

        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def attention_net(self, lstm_output, final_state):
        """
        This method computes soft alignment scores for each of the hidden_states and the last hidden_state of the LSTM.
        Tensor Sizes :
            hidden.shape = (batch_size, hidden_size)
            attn_weights.shape = (batch_size, num_seq)
            soft_attn_weights.shape = (batch_size, num_seq)
            new_hidden_state.shape = (batch_size, hidden_size)

        :param lstm_output: Final output of the LSTM which contains hidden layer outputs for each sequence.
        :param final_state: Final time-step hidden state (h_n) of the LSTM
        :return: Context vector produced by performing weighted sum of all hidden states with attention weights
        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def extract(self, input):
        input = input.transpose(0, 1)
        input = self.dropout(input)

        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.permute(1, 0, 2)

        attn_output = self.attention_net(output, final_hidden_state)
        return attn_output

    def classify(self, attn_output):
        attn_output = self.dropout2(attn_output)
        logits = self.fc(attn_output)
        return logits.squeeze(1)

    def forward(self, input):
        attn_output = self.extract(input)
        logits = self.classify(attn_output)
        return logits


class CNN(torch.nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.conv_layers = self._build_conv_layers(cfg)
        self.out_size = cfg.input_size / (cfg.pool_size**len(cfg.num_filters))
        self.flat_size = cfg.num_filters[len(cfg.num_filters)-1] * self.out_size**2
        self.fc = nn.Linear(self.flat_size, cfg.num_classes)
        self.dropout = torch.nn.Dropout(cfg.dropout)

    def _build_conv_layers(self, cfg):
        conv_layers = []
        num_channels = [1] + cfg.num_filters
        for i in range(len(num_channels)-1):
            conv_layers.append(nn.Conv2d(num_channels[i], num_channels[i+1], cfg.conv_size, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(cfg.pool_size, cfg.pool_size))
        return nn.Sequential(*conv_layers)

    def extract(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(-1, self.flat_size)
        return x

    def classify(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.classify(x)
        return x


class FeatureEnsemble(torch.nn.Module):
    def __init__(self, cfg, acoustic_model=None, linguistic_model=None):
        super(FeatureEnsemble, self).__init__()
        self.acoustic_model = acoustic_model if acoustic_model is not None else CNN(cfg.acoustic_config)
        self.linguistic_model = linguistic_model if linguistic_model is not None else AttentionLSTM(cfg.linguistic_config)
        self.feature_size = self.linguistic_model.hidden_size + self.acoustic_model.flat_size
        self.fc = nn.Linear(self.feature_size, 4)
        self.dropout = torch.nn.Dropout(0.7)


    def forward(self, input_tuple):
        acoustic_features, linguistic_features = input_tuple
        acoustic_output_features = self.acoustic_model.extract(acoustic_features)
        linguistic_output_features = self.linguistic_model.extract(linguistic_features)
        all_features = torch.cat((acoustic_output_features, linguistic_output_features), 1)
        return self.fc(self.dropout(all_features))

    @property
    def name(self):
        return "Feature Ensemble"

class LoadFeatureEnsemble(torch.nn.Module):
    def __init__(self, ensemble_model=None):
        super(LoadFeatureEnsemble, self).__init__()
        self.ensemble_model = ensemble_model
        self.linguistic_size = self.ensemble_model.linguistic_model.hidden_size
        self.acoustic_size = self.ensemble_model.acoustic_model.flat_size


    def forward(self, input_tuple):
        acoustic_features, linguistic_features = input_tuple
        acoustic_output_features = self.ensemble_model.acoustic_model.extract(acoustic_features)
        linguistic_output_features = self.ensemble_model.linguistic_model.extract(linguistic_features)
        return acoustic_output_features, linguistic_output_features

class CPMC(torch.nn.Module):
    def __init__(self):
        super(CPMC, self).__init__()
        self.fc = nn.Linear(320, 200)
        self.layer = nn.Sequential(nn.Linear(512, 320), nn.ReLU(True))

    def forward(self, input_tuple):
        x, y = input_tuple
        x = torch.unsqueeze(x, 0)
        x = self.layer(x)
        pred_y = self.fc(x)
        y = y.permute(1, 0)
        score_vector = torch.matmul(F.normalize(pred_y,dim=1), F.normalize(y,dim=0))
        return score_vector

class Classify(torch.nn.Module):
    def __init__(self):
        super(Classify, self).__init__()
        self.fc = nn.Linear(712, 4)
        self.dropout = torch.nn.Dropout(0.7)

    def forward(self, acoustic_features, linguistic_features):
        all_features = torch.cat((acoustic_features, linguistic_features), 1)
        return self.fc(self.dropout(all_features))
