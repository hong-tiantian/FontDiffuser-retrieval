import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalAdapter(nn.Module):
    """Fuse 5-slot retrieval refs into a residual for an offset-path feature map."""

    def __init__(
        self,
        feat_channels: int,
        ref_channels: int,
        n_slots: int = 5,
        slot_vocab: int = 37,
        role_vocab: int = 3,
        struct_vocab: int = 12,
        embed_dim: int = 64,
        n_heads: int = 4,
        ref_token_size: int = 12,
    ):
        super().__init__()
        if feat_channels % n_heads != 0:
            raise ValueError(
                f"feat_channels ({feat_channels}) must be divisible by n_heads ({n_heads})."
            )

        self.feat_channels = feat_channels
        self.ref_channels = ref_channels
        self.n_slots = n_slots
        self.n_heads = n_heads
        self.head_dim = feat_channels // n_heads
        self.scale = self.head_dim ** -0.5
        self.ref_token_size = ref_token_size

        self.q_norm = nn.LayerNorm(feat_channels)
        self.kv_norm = nn.LayerNorm(feat_channels)
        self.q_proj = nn.Linear(feat_channels, feat_channels)
        self.k_proj = nn.Linear(feat_channels, feat_channels)
        self.v_proj = nn.Linear(feat_channels, feat_channels)
        self.ref_in_proj = nn.Linear(ref_channels, feat_channels)

        self.slot_embed = nn.Embedding(slot_vocab, embed_dim)
        self.role_embed = nn.Embedding(role_vocab, embed_dim)
        self.struct_embed = nn.Embedding(struct_vocab, embed_dim)
        self.meta_proj = nn.Linear(embed_dim, feat_channels)

        self.attn_dropout = nn.Dropout(p=0.0)
        self.out_proj = nn.Linear(feat_channels, feat_channels)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, channels // self.n_heads)
        return x.transpose(1, 2)

    def _validate_inputs(
        self,
        h_q,
        refs,
        slot_ids,
        role_ids,
        target_struct,
        mask,
    ):
        if h_q.dim() != 4:
            raise ValueError(f"h_q must be [B, C, H, W], got {tuple(h_q.shape)}")
        if refs.dim() != 5:
            raise ValueError(f"refs must be [B, S, C, H, W], got {tuple(refs.shape)}")
        if slot_ids.dim() != 2 or role_ids.dim() != 2 or mask.dim() != 2:
            raise ValueError("slot_ids, role_ids, and mask must all be [B, S].")
        if target_struct.dim() != 1:
            raise ValueError(f"target_struct must be [B], got {tuple(target_struct.shape)}")

        batch_size, channels, _, _ = h_q.shape
        ref_batch, n_slots, ref_channels, _, _ = refs.shape
        if ref_batch != batch_size:
            raise ValueError("Batch size mismatch between h_q and refs.")
        if channels != self.feat_channels:
            raise ValueError(f"h_q channel mismatch: expected {self.feat_channels}, got {channels}.")
        if ref_channels != self.ref_channels:
            raise ValueError(
                f"refs channel mismatch: expected {self.ref_channels}, got {ref_channels}."
            )
        if n_slots != self.n_slots:
            raise ValueError(f"slot count mismatch: expected {self.n_slots}, got {n_slots}.")
        if slot_ids.shape != (batch_size, n_slots):
            raise ValueError("slot_ids shape must match [B, S].")
        if role_ids.shape != (batch_size, n_slots):
            raise ValueError("role_ids shape must match [B, S].")
        if mask.shape != (batch_size, n_slots):
            raise ValueError("mask shape must match [B, S].")
        if not mask.to(torch.bool).any(dim=1).all():
            raise ValueError(
                "RetrievalAdapter requires at least one valid slot per batch item."
            )

    def forward(
        self,
        h_q,
        refs,
        slot_ids,
        role_ids,
        target_struct,
        mask,
        return_gate=False,
    ):
        self._validate_inputs(h_q, refs, slot_ids, role_ids, target_struct, mask)

        batch_size, _, height, width = h_q.shape
        _, n_slots, _, ref_height, ref_width = refs.shape

        h_tokens = h_q.flatten(2).transpose(1, 2)
        q = self.q_proj(self.q_norm(h_tokens))

        refs_flat = refs.reshape(batch_size * n_slots, self.ref_channels, ref_height, ref_width)
        if self.ref_token_size is not None:
            refs_flat = F.adaptive_avg_pool2d(
                refs_flat, output_size=(self.ref_token_size, self.ref_token_size)
            )
        pooled_height, pooled_width = refs_flat.shape[-2:]
        ref_tokens = refs_flat.permute(0, 2, 3, 1).reshape(
            batch_size, n_slots, pooled_height * pooled_width, self.ref_channels
        )
        ref_tokens = self.ref_in_proj(ref_tokens)

        slot_bias = self.slot_embed(slot_ids)
        role_bias = self.role_embed(role_ids)
        struct_bias = self.struct_embed(target_struct).unsqueeze(1).expand(-1, n_slots, -1)
        meta_bias = self.meta_proj(slot_bias + role_bias + struct_bias).unsqueeze(2)
        ref_tokens = ref_tokens + meta_bias

        tokens_per_slot = pooled_height * pooled_width
        kv_tokens = ref_tokens.reshape(batch_size, n_slots * tokens_per_slot, self.feat_channels)
        kv_tokens = self.kv_norm(kv_tokens)
        k = self.k_proj(kv_tokens)
        v = self.v_proj(kv_tokens)

        qh = self._reshape_heads(q)
        kh = self._reshape_heads(k)
        vh = self._reshape_heads(v)

        logits = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale
        token_mask = (
            mask.to(torch.bool)
            .unsqueeze(-1)
            .expand(-1, -1, tokens_per_slot)
            .reshape(batch_size, n_slots * tokens_per_slot)
        )
        logits = logits.masked_fill(~token_mask[:, None, None, :], -1e4)

        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)
        attn_out = torch.matmul(attn, vh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, height * width, self.feat_channels
        )

        pregate = self.out_proj(attn_out)
        delta_tokens = self.alpha * pregate
        delta_h = delta_tokens.transpose(1, 2).reshape(
            batch_size, self.feat_channels, height, width
        )

        if return_gate:
            return delta_h, {
                "alpha": self.alpha,
                "pregate_norm": pregate.norm(dim=-1).mean(),
            }
        return delta_h


def attach_retrieval_adapter(
    unet,
    up_block_index: int = 2,
    feat_channels: int = 64,
    ref_channels: int = 64,
    n_slots: int = 5,
    ref_token_size: int = 12,
):
    adapter = RetrievalAdapter(
        feat_channels=feat_channels,
        ref_channels=ref_channels,
        n_slots=n_slots,
        ref_token_size=ref_token_size,
    )
    unet.up_blocks[up_block_index].retrieval_adapter = adapter
    return adapter


def freeze_backbone_train_adapter(model, up_block_index: int = 2):
    for param in model.parameters():
        param.requires_grad_(False)
    adapter = model.unet.up_blocks[up_block_index].retrieval_adapter
    if adapter is None:
        raise ValueError("No retrieval adapter attached to the requested up block.")
    for param in adapter.parameters():
        param.requires_grad_(True)
    return adapter
