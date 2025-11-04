import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *



class Model_V1(nn.Module):
    def __init__(
            self,
            nframe,
            nbin,
            nnotes,
            len_margin,
    ):
        super().__init__()

        self.nframe = nframe
        self.nbin   = nbin

        self.frameconv = FrameConv(len_margin)
        self.noteattention = NotebookAttn(nnotes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # [B, nframe, nbin]
        x = x.reshape(B*self.nframe, self.nbin)
        # [B*nframe, nbin]
        x = self.framepadding(x)
        # [B*nframe, nbin, 2M+1]
        x = self.frameconv(x)
        # [B*nframe, nbin, npitch]
        x = self.noteattention(x)
        # [B*nframe, nbin, npitch]
        return x



class Model_V1(nn.Module):
    def __init__(self, config):
        super().__init__()


    def forward(self, x):
        # [B*nframe, harmonics, nbin, 2M+1]
        # [B*nframe, ]
        return x



class FrameEncoder(nn.Module):
    def __init__(
            self,
            ch_in,
            ch_out,
            len_margin,
            n_bin:int = 216,
            n_note: int = 88,
            activation_fn:str = "leaky",
            p_dropout:float = 0.2
        ):
        super().__init__()

        if activation_fn == "relu":
            activation_layer = nn.ReLU
        elif activation_fn == "silu":
            activation_layer = nn.SiLU
        elif activation_fn == "leaky":
            activation_layer = nn.LeakyReLU
        else:
            raise ValueError

        ch_list = [32, 32, 16]

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_list[0],
                kernel_size=(15, 5),
                padding=(7, 2),
                stride=1
            ),
            activation_layer(),
            nn.Dropout(p=p_dropout)
        )

        conv_layers = []
        for i in range(len(ch_list)-1):
            conv_layers.extend([
                nn.Conv2d(
                    in_channels=ch_list[i],
                    out_channels=ch_list[i+1],
                    kernel_size=(1, 1),
                    padding=(0, 0),
                    stride=1
                ),
                activation_layer(),
                nn.Dropout(p=p_dropout)
            ])
        self.conv_layers = nn.Sequential(*conv_layers)

        self.framedown = nn.Sequential(
            nn.Conv2d(
                in_channels=ch_list[-1],
                out_channels=ch_out,
                kernel_size=(1, 2*len_margin+1),
                padding=(0, len_margin),
                stride=1
            ),
            activation_layer(),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(
                in_channels=ch_out,
                out_channels=ch_out,
                kernel_size=(1, 2*len_margin+1),
                padding=(0, 0),
                stride=1
            ),
            activation_layer(),
            nn.Dropout(p=p_dropout)
        )

        self.fc = ToeplitzLinear(n_bin, n_note)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B*nframe, ch_in, nbin, 2M+1]
        x = self.conv1(x)
        # [B*nframe, ch_list[0], nbin, 2M+1]
        x = self.conv_layers(x)
        # [B*nframe, ch_out, nbin, 2M+1]
        x = self.framedown(x)
        # [B*nframe, ch_out, nbin, 1]
        x = x.squeeze(3)
        # [B*nframe, ch_out, nbin]
        x = self.fc(x)
        # [B*nframe, ch_out, n_note]
        return x


class ToeplitzLinear(nn.Conv1d):
    def __init__(self, in_features, out_features):
        super(ToeplitzLinear, self).__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features+out_features-1,
            padding=out_features-1,
            bias=False
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super(ToeplitzLinear, self).forward(input.unsqueeze(-2)).squeeze(-2)


class Attn(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        self.query = nn.Linear(embed_size, embed_size)
        self.key   = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)
        

    def scaled_dot_product_attention(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask=None) -> torch.Tensor:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = attention_weights.masked_fill(torch.isnan(attention_weights), 0.0)

        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, embed_size = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out, _ = self.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, embed_size)

        return self.fc_out(out)