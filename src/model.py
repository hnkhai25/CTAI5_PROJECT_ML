# model.py

import torch
import torch.nn as nn
import torchvision.models as tv
from transformers import AutoProcessor, ASTModel


class ModalityDropout(nn.Module):
    """Randomly drop one modality during training."""

    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, a, v):
        if not self.training:
            return a, v
        if torch.rand(1) < self.p:
            return a, torch.zeros_like(v)
        if torch.rand(1) < self.p:
            return torch.zeros_like(a), v
        return a, v


class CrossAttentionFusion(nn.Module):
    """Audio attends to video with a residual connection."""

    def __init__(self, d_audio, d_img, d_model=256, num_heads=4, p_drop=0.1):
        super().__init__()
        self.a_proj = nn.Linear(d_audio, d_model)
        self.v_proj = nn.Linear(d_img, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.out = nn.Linear(d_model, d_model)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, a, v):
        q = self.ln_q(self.a_proj(a)).unsqueeze(1)
        kv = self.ln_kv(self.v_proj(v)).unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        out = q + self.drop(out)
        out = self.out(out.squeeze(1))
        return out


class TemporalPooling(nn.Module):
    """Pools features over the time dimension using mean and max."""

    def __init__(self, mode="meanmax"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "meanmax":
            m1 = x.mean(dim=1)
            m2 = x.max(dim=1).values
            return torch.cat([m1, m2], dim=1)
        else:
            return x.mean(dim=1)


class MultimodalEmotionRecognizer(nn.Module):
    """Main model for multimodal emotion recognition."""

    def __init__(self, num_classes, fusion, image_backbone, ast_model_id, T):
        super().__init__()
        self.T = T

        # Image Encoder
        if image_backbone == "resnet18":
            img_backbone = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
            d_img = img_backbone.fc.in_features
            img_backbone.fc = nn.Identity()
            self.visual_net = img_backbone
        else:
            raise ValueError(f"Unsupported image backbone: {image_backbone}")

        self.img_proj = nn.Linear(d_img, 128)
        self.img_norm = nn.LayerNorm(128)
        self.temporal_pool = TemporalPooling(mode="meanmax")
        d_img_proj = 256  # meanmax pooling

        # Audio Encoder
        self.processor = AutoProcessor.from_pretrained(ast_model_id)
        self.audio_net = ASTModel.from_pretrained(ast_model_id)
        d_audio = self.audio_net.config.hidden_size
        self.a_norm_in = nn.LayerNorm(d_audio)

        # Fusion
        if fusion == "crossattn":
            d_fused = 256
            self.fusion = CrossAttentionFusion(d_audio, d_img_proj, d_model=d_fused)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion}")

        self.mod_drop = ModalityDropout(p=0.15)

        # Classifier
        self.post_norm = nn.LayerNorm(d_fused)
        self.classifier = nn.Sequential(
            nn.Linear(d_fused, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = imgs.shape
        x = imgs.view(B * T, C, H, W)
        feat = self.visual_net(x)
        feat = self.img_proj(feat)
        feat = self.img_norm(feat)
        feat = feat.view(B, T, -1)
        return self.temporal_pool(feat)

    def _encode_audio_wave(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 3:
            wave = wave.squeeze(1)

        inputs = self.processor(
            wave.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(wave.device) for k, v in inputs.items()}
        outputs = self.audio_net(**inputs)
        a = outputs.last_hidden_state.mean(dim=1)
        return self.a_norm_in(a)

    def forward(self, img: torch.Tensor, wave: torch.Tensor):
        v = self._encode_images(img)
        a = self._encode_audio_wave(wave)
        a, v = self.mod_drop(a, v)
        fused = self.fusion(a, v)
        fused = self.post_norm(fused)
        return self.classifier(fused)
