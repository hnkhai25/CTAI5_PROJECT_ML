# utils.py

import torch
import torch.nn as nn
from loguru import logger


def _set_module_eval(m: nn.Module):
    """Đặt toàn bộ module (và con) về eval để tắt dropout & ngừng cập nhật BN stats."""
    # .eval() trên gốc là đủ, nhưng gọi theo modules() để chắc chắn các child cũng sync
    m.eval()
    for _ in m.modules():
        pass  # giữ chỗ; eval đã áp dụng trên gốc


def _freeze_and_eval(m: nn.Module):
    """Freeze tham số + chuyển module sang eval (ổn định khi backbone bị khoá)."""
    for p in m.parameters():
        p.requires_grad = False
    _set_module_eval(m)


def _unfreeze(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = True


def _maybe_get(obj, path_list):
    for name in path_list:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _get_ast_block_list(ast_model: nn.Module):
    # HuggingFace ASTModel
    enc = _maybe_get(ast_model, ["encoder"])
    if enc is not None:
        layers = _maybe_get(enc, ["layer", "layers"])
        if isinstance(layers, (list, nn.ModuleList)):
            return layers

    # timm ViT-like
    blocks = _maybe_get(ast_model, ["blocks"])
    if isinstance(blocks, (list, nn.ModuleList)):
        return blocks

    backbone = _maybe_get(ast_model, ["backbone"])
    if backbone is not None:
        blocks = _maybe_get(backbone, ["blocks"])
        if isinstance(blocks, (list, nn.ModuleList)):
            return blocks

    return None


def _unfreeze_last_blocks(module_list, k: int):
    if module_list is None or k <= 0:
        return
    k = min(k, len(module_list))
    for blk in module_list[-k:]:
        _unfreeze(blk)


def setup_finetune(
    model,
    img_unfreeze_last_blocks: int = 1,  # ResNet: 1 -> layer4; 2 -> layer3+4
    audio_unfreeze_last_blocks: int = 0,
):

    _freeze_and_eval(model.visual_net)
    _freeze_and_eval(model.audio_net)

    if all(
        hasattr(model.visual_net, x) for x in ["layer1", "layer2", "layer3", "layer4"]
    ):
        layers_in_order = [
            model.visual_net.layer1,
            model.visual_net.layer2,
            model.visual_net.layer3,
            model.visual_net.layer4,
        ]
        if img_unfreeze_last_blocks > 0:
            for blk in layers_in_order[-img_unfreeze_last_blocks:]:
                _unfreeze(blk)
    else:
        timm_blocks = _maybe_get(model.visual_net, ["blocks", "stages"])
        if (
            isinstance(timm_blocks, (list, nn.ModuleList))
            and img_unfreeze_last_blocks > 0
        ):
            _unfreeze_last_blocks(timm_blocks, img_unfreeze_last_blocks)
        # for name in ["conv_head", "bn2", "global_pool", "fc", "classifier"]:
        #     if hasattr(model.visual_net, name):
        #         _unfreeze(getattr(model.visual_net, name))

    ast_blocks = _get_ast_block_list(model.audio_net)
    _unfreeze_last_blocks(ast_blocks, audio_unfreeze_last_blocks)

    for name in ["layernorm", "ln_post", "norm"]:
        if hasattr(model.audio_net, name):
            _unfreeze(getattr(model.audio_net, name))

    _unfreeze(model.img_proj)
    if isinstance(model.fusion, nn.Module):
        _unfreeze(model.fusion)
    _unfreeze(model.classifier)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[Finetune] Trainable: {trainable:,} | Frozen: {frozen:,}")


def build_optimizer(model, lr_head, lr_backbone, wd_head, wd_backbone):
    head_params = (
        list(model.img_proj.parameters())
        + list(model.fusion.parameters())
        + list(model.classifier.parameters())
    )

    backbone_params = [
        p
        for p in model.parameters()
        if p.requires_grad and not any(p is pp for pp in head_params)
    ]

    param_groups = [
        {"params": head_params, "lr": lr_head, "weight_decay": wd_head},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd_backbone},
    ]

    return torch.optim.AdamW(param_groups)
