"""Minimal BRAT loader to extract query tokens from a checkpoint state dict."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import (
    DenseNet121,
    DenseNet169,
    ResNetFeatures,
    ViT,
)
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel


_DENSENET_MAP = {
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
}

_SUPPORTED_VISION_MODELS = {
    "densenet121",
    "densenet169",
    "resnet50",
    "vit",
}

class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, model_name: str, spatial_dims: int = 3):
        super().__init__()
        if model_name not in _DENSENET_MAP:
            raise ValueError(f"Unsupported DenseNet: {model_name}")
        self.densenet = _DENSENET_MAP[model_name](
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=1,
        )
        self.spatial_dims = spatial_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.densenet.features(x)
        if self.spatial_dims == 3:
            features = torch.mean(features, dim=2, keepdim=True)
        features = F.leaky_relu(features)
        features = features.view(features.shape[0], features.shape[1], -1)
        features = features.permute(0, 2, 1)
        return features


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, spatial_dims: int = 3):
        super().__init__()
        self.resnet = ResNetFeatures(
            model_name="resnet50",
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            pretrained=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.resnet(x)[-1]
        features = features.view(features.shape[0], features.shape[1], -1)
        features = features.permute(0, 2, 1)
        return features


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: tuple[int, int, int] = (32, 256, 256),
        patch_size: int = 16,
    ):
        super().__init__()
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            proj_type="conv",
            pos_embed_type="sincos",
            classification=True,
        )
        del self.vit.classification_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embedding(x)
        if hasattr(self.vit, "cls_token"):
            cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        return x[:, 1:]


def build_visual_encoder(vision_model_name: str, in_channels: int) -> nn.Module:
    if vision_model_name not in _SUPPORTED_VISION_MODELS:
        raise ValueError(f"Unsupported vision model: {vision_model_name}")

    if vision_model_name.startswith("densenet"):
        return DenseNetFeatureExtractor(in_channels, vision_model_name)
    if vision_model_name == "resnet50":
        return ResNetFeatureExtractor(in_channels)
    if vision_model_name == "vit":
        return VisionTransformerEncoder(in_channels)

    raise ValueError(f"Unsupported vision model: {vision_model_name}")


class VisionEncoderOnly(nn.Module):
    def __init__(self, vision_model_name: str, in_channels: int):
        super().__init__()
        self.visual_encoder = build_visual_encoder(vision_model_name, in_channels)

    def get_visual_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual_encoder(pixel_values)

    @classmethod
    def from_state_dict(
        cls, state_dict: dict, vision_model_name: str, in_channels: int
    ) -> "VisionEncoderOnly":
        model = cls(vision_model_name, in_channels)
        visual_state = {
            k: v for k, v in state_dict.items() if k.startswith("visual_encoder.")
        }
        model.load_state_dict(visual_state, strict=True)
        return model


class brat(nn.Module):
    def __init__(
        self,
        vision_model_name: str,
        in_channels: int,
        num_query_tokens: int,
        embed_dim: int,
        qformer_config: BertConfig,
        vision_width: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.visual_encoder = build_visual_encoder(vision_model_name, in_channels)
        self.ln_vision = nn.LayerNorm(vision_width)
        self.Qformer = BertLMHeadModel(qformer_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, qformer_config.hidden_size)
        )
        self.vision_proj = nn.Linear(qformer_config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(qformer_config.hidden_size, embed_dim)
        self.itm_head = nn.Linear(qformer_config.hidden_size, 2)
        self.temp = nn.Parameter(0.1 * torch.ones([]))

    def get_query_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeds = self.visual_encoder(pixel_values)
        image_embeds = self.ln_vision(image_embeds)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state

    @classmethod
    def from_state_dict(
        cls, state_dict: dict, vision_model_name: str, in_channels: int
    ) -> "brat":
        config = infer_qformer_config_from_state_dict(state_dict)
        model = cls(
            vision_model_name=vision_model_name,
            in_channels=in_channels,
            num_query_tokens=config["num_query_tokens"],
            embed_dim=config["embed_dim"],
            qformer_config=config["qformer_config"],
            vision_width=config["vision_width"],
        )
        model.load_state_dict(state_dict, strict=True)
        return model


def infer_qformer_config_from_state_dict(state_dict: dict) -> dict:
    vision_width = state_dict["ln_vision.weight"].shape[0]
    num_query_tokens = state_dict["query_tokens"].shape[1]
    embed_dim = state_dict["vision_proj.weight"].shape[0]

    vocab_size, hidden_size = state_dict[
        "Qformer.bert.embeddings.word_embeddings.weight"
    ].shape
    max_position_embeddings = state_dict[
        "Qformer.bert.embeddings.position_embeddings.weight"
    ].shape[0]
    token_type_key = "Qformer.bert.embeddings.token_type_embeddings.weight"
    if token_type_key in state_dict:
        type_vocab_size = state_dict[token_type_key].shape[0]
    else:
        type_vocab_size = 2
    intermediate_size = state_dict[
        "Qformer.bert.encoder.layer.0.intermediate.dense.weight"
    ].shape[0]

    layer_indices = set()
    for key in state_dict:
        if key.startswith("Qformer.bert.encoder.layer."):
            parts = key.split(".")
            if len(parts) > 4 and parts[4].isdigit():
                layer_indices.add(int(parts[4]))
    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 12

    cross_layers = sorted(
        {
            int(key.split(".")[4])
            for key in state_dict
            if "crossattention" in key and key.startswith("Qformer.bert.encoder.layer.")
        }
    )
    if len(cross_layers) >= 2:
        cross_attention_freq = cross_layers[1] - cross_layers[0]
    else:
        cross_attention_freq = 2

    encoder_width = state_dict[
        "Qformer.bert.encoder.layer.0.crossattention.self.key.weight"
    ].shape[1]

    qformer_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=12,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
    )
    qformer_config.encoder_width = encoder_width
    qformer_config.add_cross_attention = True
    qformer_config.cross_attention_freq = cross_attention_freq
    qformer_config.query_length = num_query_tokens

    return {
        "num_query_tokens": num_query_tokens,
        "embed_dim": embed_dim,
        "vision_width": vision_width,
        "qformer_config": qformer_config,
    }
