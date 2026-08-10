import glob
import logging
import os
import subprocess
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from robustbench import load_model

from common.madry_imagenet_l2 import is_madry_l2_model, load_madry_l2_model

warnings.filterwarnings("ignore")

logging.getLogger("open_clip").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# -------------------------
# VLM Wrapper
# -------------------------
class VLMZeroShotWrapper(nn.Module):
    def __init__(self, model, text_embedding):
        super().__init__()
        self.model = model
        self.register_buffer("text_embedding", text_embedding)

    def forward(self, vision):
        image_features = self.model.encode_image(vision, normalize=True)

        current_device = image_features.device
        aligned_text_embedding = self.text_embedding.to(current_device)
        logits = image_features @ aligned_text_embedding

        logit_scale = self.model.logit_scale.to(current_device).exp()
        logits *= logit_scale
        
        return logits

# -------------------------
# Model loading helpers
# -------------------------
class NormalizeWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mean, std):
        super().__init__()
        self.model = model
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.model(x)

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _checkpoint_state_dict(weight_path: str) -> dict:
    ckpt = torch.load(weight_path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)


def _is_full_openclip_state(state: dict) -> bool:
    keys = state.keys()
    has_vision = any(k.startswith("visual.") for k in keys)
    has_text = "token_embedding.weight" in state or any(
        k.startswith("transformer.resblocks.") for k in keys
    )
    return has_vision and has_text


def _visual_pos_dim(state: dict) -> Optional[int]:
    pos = state.get("visual.positional_embedding")
    if pos is not None:
        return int(pos.shape[-1])
    pos = state.get("positional_embedding")
    if pos is not None and int(pos.shape[0]) > 100:
        return int(pos.shape[-1])
    return None


def _validate_vlm_checkpoint(state: dict, arch: str, weight_path: str) -> None:
    dim = _visual_pos_dim(state)
    if dim is None:
        return
    if arch.startswith("ViT-B") and dim >= 1024:
        raise RuntimeError(
            f"{weight_path} looks like ViT-L/14 weights (hidden={dim}), not {arch}. "
            "Delete the file and re-run: bash scripts/download_vlm_weights.sh"
        )
    if arch.startswith("ViT-L") and dim <= 768:
        raise RuntimeError(
            f"{weight_path} looks like ViT-B/16 weights (hidden={dim}), not {arch}. "
            "Delete the file and re-run: bash scripts/download_vlm_weights.sh"
        )


def _load_visual_checkpoint(model: nn.Module, state: dict) -> None:
    prefix = "visual."
    visual_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model.visual.load_state_dict(visual_state or state, strict=False)


def _load_vlm_openclip(*, arch: str, base_tag: Optional[str], weight_path: str):
    import open_clip

    state = _checkpoint_state_dict(weight_path)
    _validate_vlm_checkpoint(state, arch, weight_path)
    abs_path = os.path.abspath(weight_path)

    if _is_full_openclip_state(state):
        # HF open_clip_pytorch_model.bin (saved as .pt): load locally, no Hub download.
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=arch, pretrained=abs_path
            )
            return model, _, preprocess
        except Exception:
            pass
        # Fallback: raw OpenCLIP state dict (e.g. leaf_vit_l.pt from convert script).
        init_tag = base_tag if base_tag else None
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=arch, pretrained=init_tag
        )
        model.load_state_dict(state, strict=False)
        return model, _, preprocess

    if base_tag is None:
        raise ValueError(
            f"Visual-only checkpoint {weight_path} needs base_tag to assemble {arch}"
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=arch, pretrained=base_tag
    )
    _load_visual_checkpoint(model, state)
    return model, _, preprocess


def _vlm_text_embed_cache_path(
    weight_dir: str,
    model_name: str,
    weight_path: Optional[str] = None,
) -> str:
    cache_dir = os.path.join(weight_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe = model_name.replace(os.sep, "_").replace(" ", "_")
    if weight_path and os.path.isfile(weight_path):
        tag = f"w{int(os.path.getmtime(weight_path))}"
    else:
        tag = "online"
    return os.path.join(cache_dir, f"text_embed_{safe}_{tag}_zs80.pt")


def _vlm_encode_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _imagenet_zero_shot_classnames() -> List[str]:
    """CLIP / RobustVLM ImageNet-1K class names (not torchvision scientific names)."""
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES

    return list(IMAGENET_CLASSNAMES)


def _imagenet_zero_shot_templates():
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES

    return OPENAI_IMAGENET_TEMPLATES


def _encode_vlm_text_embeddings(
    model: nn.Module,
    encode_device: torch.device,
) -> torch.Tensor:
    """Build [embed_dim, 1000] zero-shot weights (80 templates per class, mean-pooled)."""
    import open_clip

    classnames = _imagenet_zero_shot_classnames()
    templates = _imagenet_zero_shot_templates()
    model = model.to(encode_device).eval()
    weights: List[torch.Tensor] = []
    with torch.no_grad():
        for classname in classnames:
            texts = [template(classname) for template in templates]
            tokens = open_clip.tokenize(texts).to(
                encode_device, non_blocking=encode_device.type == "cuda"
            )
            feats = model.encode_text(tokens, normalize=True)
            class_embed = feats.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            weights.append(class_embed)
    return torch.stack(weights, dim=1).contiguous()


def _vlm_text_embeddings(
    model: nn.Module,
    *,
    model_name: str,
    weight_dir: str,
    weight_path: Optional[str] = None,
) -> torch.Tensor:
    encode_device = _vlm_encode_device()
    cache_path = _vlm_text_embed_cache_path(weight_dir, model_name, weight_path)
    if os.path.isfile(cache_path):
        return torch.load(cache_path, map_location=encode_device)

    embedding = _encode_vlm_text_embeddings(model, encode_device)
    torch.save(embedding.detach().cpu(), cache_path)
    return embedding


def load_one_model(dataset: str, name: str, threat_model: str = "Linf") -> torch.nn.Module:
    if dataset == "cifar10":
        if name == "R56_cifar10":
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True).eval()
            return NormalizeWrapper(model, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).eval()
        return load_model(name, dataset="cifar10", threat_model=threat_model).eval()

    if dataset == "imagenet":
        is_vlm = any(kw in name for kw in ["CLIP", "TeCoA", "FARE", "SimCLIP", "LEAF"])
        
        if is_vlm:
            import open_clip

            weight_dir = os.path.join(_repo_root(), "checkpoints", "vlm_weights")

            if name == "ViT-L-14_TeCoA4":
                weight_path, arch, base_tag = f"{weight_dir}/tecoa_eps_4.pt", "ViT-L-14-quickgelu", None
            elif name == "ViT-L-14_FARE4":
                weight_path, arch, base_tag = f"{weight_dir}/fare_eps_4.pt", "ViT-L-14-quickgelu", None
            elif name == "ViT-L-14_SimCLIP4":
                # simclip4.pt is visual-encoder only (fine-tuned on OpenAI CLIP ViT-L/14)
                weight_path, arch, base_tag = f"{weight_dir}/simclip4.pt", "ViT-L-14-quickgelu", "openai"
            elif name == "ViT-L-14_LEAF-CLIP":
                weight_path, arch, base_tag = f"{weight_dir}/leaf_vit_l.pt", "ViT-L-14-quickgelu", None
            elif name == "ViT-L-14_CLIP":
                weight_path, arch, base_tag = None, "ViT-L-14", "openai"

            elif name == "ConvNeXt-B_RCLIP-TeCoA":
                weight_path, arch, base_tag = f"{weight_dir}/convnext_base_w-tecoa-eps4.pt", "convnext_base_w", "laion2b_s13b_b82k_augreg"
            elif name == "ConvNeXt-B_RCLIP-FARE":
                weight_path, arch, base_tag = f"{weight_dir}/convnext_base_w-fare-eps4.pt", "convnext_base_w", "laion2b_s13b_b82k_augreg"

            elif name == "ViT-B-16_RCLIP-TeCoA":
                weight_path, arch, base_tag = f"{weight_dir}/vit-b-16-tecoa-eps4.pt", "ViT-B-16", "laion2b_s34b_b88k"
            elif name == "ViT-B-16_RCLIP-FARE":
                weight_path, arch, base_tag = f"{weight_dir}/vit-b-16-fare-eps4.pt", "ViT-B-16", "laion2b_s34b_b88k"
                
            else:
                raise ValueError(f"UNKNOWN VLM MODEL ID: {name}")

            if weight_path and not os.path.exists(weight_path):
                download_script = os.path.join(_repo_root(), "scripts", "download_vlm_weights.sh")
                subprocess.run(["bash", download_script], check=True, cwd=_repo_root())

            if weight_path:
                model, _, _ = _load_vlm_openclip(
                    arch=arch, base_tag=base_tag, weight_path=weight_path
                )
            else:
                model, _, _ = open_clip.create_model_and_transforms(
                    model_name=arch, pretrained=base_tag
                )

            model = model.eval()
            encode_device = _vlm_encode_device()
            embedding_text_labels_norm = _vlm_text_embeddings(
                model,
                model_name=name,
                weight_dir=weight_dir,
                weight_path=weight_path,
            )
            model = model.to(encode_device)
            zero_shot_model = VLMZeroShotWrapper(model, embedding_text_labels_norm)
            clip_mean, clip_std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
            return NormalizeWrapper(zero_shot_model, clip_mean, clip_std).eval()

        if name == "ViT-B":
            return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000).eval()
        if name == "ConvNeXt-B":
            return timm.create_model("convnext_base.fb_in1k", pretrained=True, num_classes=1000).eval()
        if name == "Swin-B":
            return timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=1000).eval()

        if is_madry_l2_model(name):
            return load_madry_l2_model(name)

        return load_model(name, dataset="imagenet", threat_model=threat_model).eval()

    raise ValueError(f"Unsupported dataset: {dataset!r}")


def dataset_spatial_size(dataset: str) -> Tuple[int, int]:
    if dataset == "cifar10":
        return 32, 32
    if dataset in ("imagenet", "celeba"):
        return 224, 224
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def maybe_resize_samples(x: np.ndarray, dataset: str) -> np.ndarray:
    target_h, target_w = dataset_spatial_size(dataset)
    if x.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W] array, got shape {x.shape}")
    _, _, h, w = x.shape
    if h == target_h and w == target_w:
        return x

    print(
        f"Resizing samples {h}x{w} -> {target_h}x{target_w} for --task {dataset} "
        f"({x.shape[0]} images)...",
        flush=True,
    )
    t = torch.from_numpy(np.ascontiguousarray(x)).float()
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    t = torch.nn.functional.interpolate(
        t, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    if x.shape[1] == 1:
        t = t[:, :1]
    out = t.numpy()
    print(f"Resize finished ({out.shape[0]} images).", flush=True)
    return np.ascontiguousarray(out)


def _sample_filepath(sample_dir: str, index: int) -> str:
    """Resolve 00042_label.png without scanning the whole directory."""
    matches = sorted(glob.glob(os.path.join(sample_dir, f"{index:05d}_*")))
    if not matches:
        raise FileNotFoundError(f"No image for index {index:05d} under {sample_dir}")
    return matches[0]


def load_samples(
    path: str,
    start_idx: Optional[int] = 0,
    end_idx: Optional[int] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Any, Any]:
    samplepath = os.path.join(r"", f"{path}")
    if end_idx is None:
        matches = glob.glob(os.path.join(samplepath, "[0-9][0-9][0-9][0-9][0-9]_*"))
        if not matches:
            raise FileNotFoundError(f"No indexed samples under {path}")
        end_idx = max(int(os.path.basename(m)[:5]) for m in matches) + 1

    if end_idx <= start_idx:
        raise ValueError(f"end_idx ({end_idx}) must be > start_idx ({start_idx})")

    n_load = end_idx - start_idx
    if n_load >= 50:
        print(f"Loading {n_load} samples [{start_idx},{end_idx}) from {path}...", flush=True)

    images: List[np.ndarray] = []
    labels: List[int] = []
    for i in range(start_idx, end_idx):
        filepath = _sample_filepath(samplepath, i)
        label = int(os.path.basename(filepath).split(".")[0].split("_")[-1])
        with Image.open(filepath) as pil_img:
            if shape is not None:
                pil_img = pil_img.resize(shape)
            image = np.asarray(pil_img, dtype=np.float32)
        if image.ndim == 2:
            image = image[..., np.newaxis]
        assert image.ndim == 3
        images.append(np.transpose(image, (2, 0, 1)))
        labels.append(label)

    images_ = np.stack(images)
    labels_ = np.array(labels)
    images_ = np.ascontiguousarray(images_ / 255.0)
    return images_, labels_


def chunk_tensor(x: torch.Tensor, bs: int) -> List[torch.Tensor]:
    if bs <= 0:
        raise ValueError("batch_size must be positive")
    return [x[i : i + bs] for i in range(0, x.shape[0], bs)]
