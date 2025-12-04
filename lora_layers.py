import torch.nn.functional as F
import torch.nn as nn
import torch


class LoRABaseLayer:
    """
    Ancestor in order to define shared parameters among the different LoRA layers.
    """

    def __init__(self, rank=8, alpha=8, dropout=0.0):
        self.rank = rank
        self.alpha = alpha
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def load_pretrained_parameters(self, state_dict):
        if "bias" in state_dict.keys():
            self.bias.data = state_dict["bias"]

        self.weight.data = state_dict["weight"]


class LoRALinearLayer(LoRABaseLayer, nn.Linear):

    def __init__(self, in_features, out_features, bias=True, rank=8, alpha=16, dropout=0.0, **kwargs):

        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        LoRABaseLayer.__init__(self, rank=rank, alpha=alpha, dropout=dropout)

        # Turn-off gradient
        self.weight.requires_grad = False

        # LoRA matrix B initialized with 0 weights
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # LoRA matrix A initialized with Gaussian normal distribution
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)

    def sum_weights(self):
        """
        xW + xAB = x(W + AB) This function sum the pre-trained weights and AB matrix multiplication.
        Thus, inference is faster on runtime.
        """

        summed_weight = self.weight.data + self.scale * torch.matmul(self.lora_A, self.lora_B).T

        # Save parameters
        state_dict = {"weight": summed_weight}
        if self.bias is not None:
            state_dict["bias"] = self.bias

        # Load final tuned-weights to a linear torch layer
        linear = nn.Linear(self.in_features, self.out_features, bias=True if self.bias is not None else False)
        linear.load_state_dict(state_dict)

        return linear

    def forward(self, x):
        # Pre-trained weights
        pretrained_out = F.linear(x, self.weight, bias=self.bias)

        # LoRA weights
        lora_mul = self.scale * torch.matmul(self.lora_A, self.lora_B)
        lora_out = torch.matmul(self.dropout(x), lora_mul)

        # Sum multiplication results
        return pretrained_out + lora_out


class LoRAEmbeddingLayer(nn.Embedding, LoRABaseLayer):

    def __init__(self, num_embeddings, embedding_dim, rank=8, alpha=16, dropout=0.0, **kwargs):

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRABaseLayer.__init__(self, rank=rank, alpha=alpha, dropout=dropout)

        # Turn-off gradient
        self.weight.requires_grad = False

        # LoRA matrix B initialized with 0 weights
        self.lora_B = nn.Parameter(torch.zeros(rank, self.embedding_dim))

        # LoRA matrix A initialized with Gaussian normal distribution
        self.lora_A = nn.Parameter(torch.zeros(self.num_embeddings, rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)

    def _merge_weights(self):
        """
        xW + xAB = x(W + AB) This function sum the pre-trained weights and AB matrix multiplication.
        Thus, inference is faster on runtime.
        """

        summed_weights = self.weight.data + self.scale * torch.matmul(self.lora_A, self.lora_B).T

        # Save parameters
        state_dict = {"weight": summed_weights}
        embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        embedding.load_state_dict(state_dict)

        return embedding

    def forward(self, x):

        # Pre-trained weights
        pretrained_out = F.embedding(input=x,
                                     weight=self.weight,
                                     padding_idx=self.padding_idx,
                                     max_norm=self.max_norm,
                                     norm_type=self.norm_type,
                                     scale_grad_by_freq=self.scale_grad_by_freq,
                                     sparse=self.sparse)

        # A weights
        a_out = F.embedding(input=x,
                                        weight=self.lora_A,
                                        padding_idx=self.padding_idx,
                                        max_norm=self.max_norm,
                                        norm_type=self.norm_type,
                                        scale_grad_by_freq=self.scale_grad_by_freq,
                                        sparse=self.sparse)

        # LoRA weights
        ab_out = self.scale * torch.matmul(a_out, self.lora_B)

        return pretrained_out + ab_out

