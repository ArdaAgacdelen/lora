from dataclasses import dataclass
from typing import Optional, Literal, Union
from safetensors.torch import save_file
import torch.nn as nn

from lora_layers import LoRAEmbeddingLayer, LoRALinearLayer


@dataclass
class LoraConfig:
    rank: int = 8
    bias: Literal["none", "all", "lora_biases"] = "none"
    alpha: float = 16.0
    dropout: float = 0.0

    modelComponents = Literal["word_embeddings", "query", "key", "value", "dense", "classifier"]
    target_modules: Optional[Union[modelComponents, list[modelComponents]]] = None
    exclude_modules: Optional[Union[list[str], str]] = None


class LoraModel(nn.Module):

    def __init__(self, model, config):

        super(LoraModel, self).__init__()

        self.model = model
        self.config = config

        if self.config.target_modules is None:
            self.config.target_modules = []
        elif isinstance(self.config.target_modules, str):
            self.config.target_modules = [self.config.target_modules]

        if self.config.exclude_modules is None:
            self.config.exclude_modules = []
        elif isinstance(self.config.exclude_modules, str):
            self.config.exclude_modules = [self.config.exclude_modules]

        # Number of learnable parameters
        self.n_pretrained = self.get_n_trainable()

        # Turn-off all gradients
        self._disable_all_grads()

        # Make the model LoRA compatible
        self._make_lora(self.model)

        # Handle biases' gradients
        self._handle_bias_grad()


    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def _is_exclude_module(self, name):
        return any([ex in name for ex in self.config.exclude_modules])

    def _is_target_module(self, name):
        return any([tgt in name for tgt in self.config.target_modules])

    def _make_lora(self, module):
        """
        Recursively replace all the layers in a model with lora_layers
        """

        for name, child in module.named_children():

            if self._is_target_module(name):

                if isinstance(child, nn.Linear):

                    new_layer = LoRALinearLayer(in_features=child.in_features,
                                                out_features=child.out_features,
                                                bias=True if child.bias is not None else False,
                                                rank=self.config.rank,
                                                alpha=self.config.alpha,
                                                dropout=self.config.dropout)

                    new_layer.load_pretrained_parameters(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Embedding):

                    new_layer = LoRAEmbeddingLayer(num_embeddings=child.num_embeddings,
                                                   embedding_dim=child.embedding_dim,
                                                   rank=self.config.rank,
                                                   alpha=self.config.alpha,
                                                   dropout=self.config.dropout)

                    new_layer.load_pretrained_parameters(child.state_dict())
                    setattr(module, name, new_layer)

            if (len(list(child.children())) > 0) and not any([ex in name for ex in self.config.exclude_modules]):
                self._make_lora(child)

    def _handle_bias_grad(self):
        """
        Turn off bias gradients depending on:
            - none:  Dont train any biases
            - all: train all biases
            - lora_biases: train biases only in lora layers
        """

        for name, param in self.model.named_parameters():

            if not self._is_exclude_module(name):
                if ".bias" in name:
                    if self.config.bias == "none":
                        param.requires_grad = False
                    elif self.config.bias == "all":
                        param.requires_grad = True
                    elif (self.config.bias == "lora_biases") and self._is_target_module(name):
                        param.requires_grad = True

    def _disable_all_grads(self):

        for name, param in self.model.named_parameters():

            if not self._is_exclude_module(name):
                param.requires_grad = False

    def get_n_trainable(self):
        total_learnable_params = 0
        for param in self.model.parameters():
            if param.requires_grad:
                total_learnable_params += param.numel()

        return total_learnable_params

    def _merge_weights(self, module):

        """
        Recursively sum pre-trained and (AB) weights and replace LoRA layers with torch.Linear layers in model.
        """

        for name, child in module.named_children():

            if isinstance(child, (LoRALinearLayer, LoRAEmbeddingLayer)):

                merged_layer = child.sum_weights()

                setattr(module, name, merged_layer)

            else:

                if len(list(child.children())) > 0:
                    self._merge_weights(child)

    def save_model(self, path, merge_weights=False):
        """
        Save model safetensors to the given path
            - merge_weights: True = Merge LoRA weights and save
            - merge_weights: False = Only save trainable weights
        """

        def _detach_cpu(param):
            return param.detach().cpu()

        if merge_weights:

            self._merge_weights(self.model)

            state_dict = {name.replace("lora_model.", ""): _detach_cpu(param) for (name, param) in
                          self.named_parameters()}

        else:
            state_dict = {name: _detach_cpu(param) for (name, param) in self.named_parameters() if param.requires_grad}

        save_file(state_dict, path)
