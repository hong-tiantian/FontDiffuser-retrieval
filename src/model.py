import math
import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)

class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    # --- CALLI-RAG BEGIN: convert 5-slot retrieval images into content-encoder features ---
    @staticmethod
    def _select_ref_residual_feature(residual_features, ref_channels: int = 64):
        for feature in residual_features:
            if feature.shape[1] == ref_channels:
                return feature
        shapes = [tuple(feature.shape) for feature in residual_features]
        raise ValueError(
            f"content_encoder did not return a {ref_channels}-channel residual map; got {shapes}."
        )

    def _build_retrieval_unet_inputs(self, retrieval_inputs, content_images):
        if retrieval_inputs is None:
            return None

        unet_inputs = {}
        for key in ("slot_ids", "role_ids", "target_struct", "mask"):
            if key not in retrieval_inputs:
                raise KeyError(f"retrieval_inputs missing required key: {key}")
            value = retrieval_inputs[key]
            unet_inputs[key] = value.to(content_images.device) if torch.is_tensor(value) else value

        if "refs" in retrieval_inputs:
            refs = retrieval_inputs["refs"]
            unet_inputs["refs"] = refs.to(device=content_images.device, dtype=content_images.dtype)
            return unet_inputs

        if "ref_images" not in retrieval_inputs:
            raise KeyError("retrieval_inputs must contain either 'refs' or 'ref_images'.")

        ref_images = retrieval_inputs["ref_images"].to(
            device=content_images.device,
            dtype=content_images.dtype,
        )
        if ref_images.dim() != 5:
            raise ValueError(
                f"ref_images must be [B, 5, 3, H, W], got {tuple(ref_images.shape)}"
            )

        batch_size, n_slots, channels, height, width = ref_images.shape
        ref_images_flat = ref_images.reshape(batch_size * n_slots, channels, height, width)
        with torch.no_grad():
            _, ref_residual_features = self.content_encoder(ref_images_flat)
        ref_feature = self._select_ref_residual_feature(ref_residual_features)
        unet_inputs["refs"] = ref_feature.reshape(
            batch_size,
            n_slots,
            ref_feature.shape[1],
            ref_feature.shape[2],
            ref_feature.shape[3],
        )
        return unet_inputs
    # --- CALLI-RAG END ---
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        style_images,
        content_images,
        content_encoder_downsample_size,
        # --- CALLI-RAG BEGIN: optional retrieval adapter inputs ---
        retrieval_inputs=None,
        # --- CALLI-RAG END ---
    ):
        style_img_feature, _, _ = self.style_encoder(style_images)
    
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
    
        # Get the content feature
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, \
                               style_hidden_states, style_content_res_features]
        # --- CALLI-RAG BEGIN: prepare retrieval features for UNet ---
        retrieval_unet_inputs = self._build_retrieval_unet_inputs(
            retrieval_inputs=retrieval_inputs,
            content_images=content_images,
        )
        # --- CALLI-RAG END ---

        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
            # --- CALLI-RAG BEGIN: pass optional retrieval features ---
            retrieval_inputs=retrieval_unet_inputs,
            # --- CALLI-RAG END ---
        )
        noise_pred = out[0]
        offset_out_sum = out[1]
        
        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """
    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        cond,
        content_encoder_downsample_size,
        version,
    ):
        content_images = cond[0]
        style_images = cond[1]

        style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        # Get content feature
        content_img_feture, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feture)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states, style_content_res_features]

        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        
        return noise_pred
