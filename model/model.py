# model/model.py
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence length T and d_model dims.
    Input: (T, batch, d_model) expected by nn.TransformerEncoder, but we'll adapt when used.
    """

    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd case
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (T, batch, d_model) -> returns x + pe[:T]
        """
        T = x.size(0)
        x = x + self.pe[:T]
        return x


class CNN_BiLSTM_Transformer(nn.Module):
    def __init__(self,
                 n_features,
                 cnn_channels=64,
                 cnn_kernel_size=3,
                 lstm_hidden=128,
                 lstm_layers=1,
                 transformer_dmodel=128,
                 transformer_nhead=4,
                 transformer_layers=2,
                 transformer_dim_feedforward=256,
                 num_classes=7,
                 dropout=0.2):
        """
        n_features: input feature dimension F
        Input sequence shape: (batch, T, F)
        Pipeline:
          - CNN (1D over time): input (batch, F, T) => conv layers => (batch, C, T')
          - transpose => (T', batch, C)
          - BiLSTM => (T', batch, 2*hidden)
          - Project to transformer_dmodel and feed to TransformerEncoder
          - Global average pooling over time and final linear classifier
        """
        super().__init__()
        self.n_features = n_features

        # --- CNN block: conv1d across time axis with input channels = n_features
        # We'll use a small stack of conv layers to get per-time-step feature extraction.
        # Input to Conv1d must be (batch, in_channels, seq_len) where in_channels = n_features
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=cnn_channels,
                      kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                      kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
        )

        # --- BiLSTM: input size = cnn_channels, outputs hidden dim
        self.bilstm = nn.LSTM(input_size=cnn_channels,
                              hidden_size=lstm_hidden,
                              num_layers=lstm_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=dropout if lstm_layers > 1 else 0.0)

        # Project BiLSTM output dim (2*lstm_hidden) to transformer d_model
        self.project_to_transformer = nn.Linear(
            2 * lstm_hidden, transformer_dmodel)

        # Positional encoding + TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dmodel,
                                                   nhead=transformer_nhead,
                                                   dim_feedforward=transformer_dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers)
        self.pos_encoder = PositionalEncoding(transformer_dmodel, max_len=2000)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dmodel),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dmodel, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, T, F)
        returns logits: (batch, num_classes)
        """
        batch, T, F = x.shape
        # CNN expects (batch, in_channels, seq_len) where in_channels = F
        x_c = x.permute(0, 2, 1)   # (batch, F, T)
        c_out = self.cnn(x_c)      # (batch, C, T) (same T due to padding)
        c_out = c_out.permute(0, 2, 1)  # (batch, T, C)

        # BiLSTM expects (batch, seq_len, input_size)
        lstm_out, _ = self.bilstm(c_out)  # (batch, T, 2*lstm_hidden)

        # Project to transformer d_model
        proj = self.project_to_transformer(lstm_out)  # (batch, T, d_model)

        # Transformer expects (seq_len, batch, d_model)
        proj = proj.permute(1, 0, 2)  # (T, batch, d_model)
        proj = self.pos_encoder(proj)
        trans_out = self.transformer(proj)  # (T, batch, d_model)

        # Pool over time (mean)
        trans_out = trans_out.permute(1, 0, 2)  # (batch, T, d_model)
        pooled = trans_out.mean(dim=1)          # (batch, d_model)

        logits = self.classifier(pooled)        # (batch, num_classes)
        return logits
