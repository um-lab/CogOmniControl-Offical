# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# https://github.com/bmaltais/kohya_ss

import hashlib
import math
import os
from collections import defaultdict
from io import BytesIO
from typing import List, Optional, Type, Union

import safetensors.torch
import torch
import torch.utils.checkpoint
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from safetensors.torch import load_file
from transformers import T5EncoderModel


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x, *args, **kwargs):
        weight_dtype = x.dtype
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x.to(self.lora_down.weight.dtype))

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded.to(weight_dtype) + lx.to(weight_dtype) * self.multiplier * scale


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


class LoRANetwork(torch.nn.Module):
    TRANSFORMER_TARGET_REPLACE_MODULE = ["CogVideoXTransformer3DModel", "WanTransformer3DModel", "VaceWanModel", "Wan2_2Transformer3DModel", "CogOmniControlWanModel", "CogOmniControlWanModel_Connector", "CogOmniControlWanModel_Extra_Connector"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["T5LayerSelfAttention", "T5LayerFF", "BertEncoder", "T5SelfAttention", "T5CrossAttention"]
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    def __init__(
        self,
        text_encoder: Union[List[T5EncoderModel], T5EncoderModel],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        skip_names: Optional[List[str]] = None,
        varbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout

        print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
        print(f"neuron dropout: p={self.dropout}")

        # create module instances
        def create_modules(
            is_unet: bool,
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
        ) -> List[LoRAModule]:
            prefix = (
                self.LORA_PREFIX_TRANSFORMER
                if is_unet
                else self.LORA_PREFIX_TEXT_ENCODER
            )
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "LoRACompatibleLinear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d" or child_module.__class__.__name__ == "LoRACompatibleConv"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
                        
                        if skip_names is not None:
                            skip_module = False
                            for skip_name in skip_names:
                                if skip_name in child_name:
                                    skip_module = True
                                    break
                            if skip_module:
                                print(f"skipping {child_name} lora module.")
                                continue

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha = None

                            if is_linear or is_conv2d_1x1:
                                dim = self.lora_dim
                                alpha = self.alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1:
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                            )
                            loras.append(lora)
            return loras, skipped

        text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]

        self.text_encoder_loras = []
        skipped_te = []
        for i, text_encoder in enumerate(text_encoders):
            if text_encoder is not None:
                text_encoder_loras, skipped = create_modules(False, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
                self.text_encoder_loras.extend(text_encoder_loras)
                skipped_te += skipped
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras, skipped_un = create_modules(True, unet, LoRANetwork.TRANSFORMER_TARGET_REPLACE_MODULE)
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True, add_connector=False, add_extra_layer=False):
        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if add_connector:
            self.add_module("llm_fea_connector", unet.llm_fea_connector)

        if add_extra_layer:
            for name, module in unet.named_modules():
                if name.endswith('_extra'):
                    self.add_module(name.replace(".", "_"), module)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        info = self.load_state_dict(weights_sd, False)
        return info


    def load_connector(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        new_weights_sd = {
            k.replace("llm_fea_connector.", ""): v
            for k, v in weights_sd.items()
            if k.startswith("llm_fea_connector.")
        }
        info = self.llm_fea_connector.load_state_dict(new_weights_sd, False)
        return info

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data["lr"] = unet_lr
            all_params.append(param_data)

        return all_params
    
    def prepare_joint_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)

        self.llm_fea_connector.requires_grad_(True)

        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data["lr"] = unet_lr
            all_params.append(param_data)

        param_data = {"params": enumerate_params(self.llm_fea_connector)}
        param_data["lr"] = unet_lr
        all_params.append(param_data)

        return all_params

    def prepare_joint_extra_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)

        self.llm_fea_connector.requires_grad_(True)

        for name, module in self.named_modules():
            if name.endswith('_extra'):
                module.requires_grad_(True)

        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data["lr"] = unet_lr
            all_params.append(param_data)

        param_data = {"params": enumerate_params(self.llm_fea_connector)}
        param_data["lr"] = unet_lr
        all_params.append(param_data)

        params = []
        for name, module in self.named_modules():
            if name.endswith('_extra'):
                params.extend(module.parameters())

        param_data = {"params": params}
        param_data["lr"] = unet_lr
        all_params.append(param_data)


        return all_params

    def enable_gradient_checkpointing(self):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    text_encoder: Union[T5EncoderModel, List[T5EncoderModel]],
    transformer,
    neuron_dropout: Optional[float] = None,
    skip_names: Optional[List[str]] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    network = LoRANetwork(
        text_encoder,
        transformer,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        skip_names=skip_names,
        varbose=True,
    )
    return network


def format_lora_state_dict(state_dict):
    """
    将标准 Diffusers 格式转换为 VideoX-Fun/Kohya 扁平化格式。
    【新增 Diff 兼容】不仅保留 lora_up/down，也放行 .diff 系列全秩残差权重。
    """
    new_state_dict = {}
    
    # 定义合法的 LoRA/Diff 后缀白名单
    valid_suffixes = [
        ".lora_down.weight", 
        ".lora_up.weight", 
        ".alpha",
        ".diff",    # 新增：直接相加的权重残差
        ".diff_b",  # 新增：直接相加的偏置残差
        ".diff_m"   # 新增：调制层残差
    ]

    for key, value in state_dict.items():
        # 识别后缀
        suffix = None
        stem = key
        
        for s in valid_suffixes:
            if key.endswith(s):
                suffix = s
                stem = key[:-len(s)]
                break
        
        # 过滤：如果没有识别到标准后缀，直接跳过
        if suffix is None:
            continue

        # 【重点】过滤逻辑优化：如果含有 'norm'，但属于合法的 diff（比如 norm.diff），则放行！
        # 只有当它是 norm 且不是 diff 时（比如某些炼废了的常规 norm LoRA），才拦截。
        is_diff_type = suffix in [".diff", ".diff_b", ".diff_m"]
        if "norm" in key and not is_diff_type:
            continue

        # 核心转换逻辑
        if "diffusion_model." in stem:
            content = stem.replace("diffusion_model.", "")
            content = content.replace(".", "_") # 将路径中的点换成下划线
            
            new_key = f"lora_unet__{content}{suffix}"
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def merge_lora(pipeline, lora_path, multiplier, device='cpu', dtype=torch.float32, state_dict=None, transformer_only=False, sub_transformer_name="transformer"):
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if state_dict is None:
        state_dict = load_file(lora_path)
    else:
        state_dict = state_dict
        
    if LORA_PREFIX_TRANSFORMER not in next(iter(state_dict.keys())):
        state_dict = format_lora_state_dict(state_dict)
    
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    sequential_cpu_offload_flag = False
    if pipeline.transformer.device == torch.device(type="meta"):
        pipeline.remove_all_hooks()
        sequential_cpu_offload_flag = True
        offload_device = pipeline._offload_device

    for layer, elems in updates.items():

        if "lora_te" in layer:
            if transformer_only:
                continue
            else:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
            curr_layer = getattr(pipeline, sub_transformer_name)

        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name + "_" + "_".join(layer_infos))
                    break
                except Exception:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name)
                        if len(layer_infos) > 0:
                            temp_name = layer_infos.pop(0)
                        elif len(layer_infos) == 0:
                            break
                    except Exception:
                        if len(layer_infos) == 0:
                            print('Error loading layer')
                        if len(temp_name) > 0:
                            temp_name += "_" + layer_infos.pop(0)
                        else:
                            temp_name = layer_infos.pop(0)

        # 获取原始层类型和设备信息
        origin_dtype = curr_layer.weight.data.dtype if hasattr(curr_layer, "weight") else curr_layer.data.dtype
        origin_device = curr_layer.weight.data.device if hasattr(curr_layer, "weight") else curr_layer.data.device

        curr_layer = curr_layer.to(device, dtype)

        # ---------------------------------------------------------
        # 【新增】混合处理逻辑：支持常规矩阵乘法与 Diff 直接相加
        # ---------------------------------------------------------

        # 1. 处理标准的 Low-Rank (up + down)
        if 'lora_up.weight' in elems and 'lora_down.weight' in elems:
            weight_up = elems['lora_up.weight'].to(device, dtype)
            weight_down = elems['lora_down.weight'].to(device, dtype)
            
            if 'alpha' in elems.keys():
                alpha = elems['alpha'].item() / weight_up.shape[1]
            else:
                alpha = 1.0

            if len(weight_up.shape) == 4:
                lora_delta = torch.mm(
                    weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                lora_delta = torch.mm(weight_up, weight_down)
                
            curr_layer.weight.data += multiplier * alpha * lora_delta

        # 2. 处理加速 LoRA 的全秩残差 (Diff)
        if 'diff' in elems:
            diff_weight = elems['diff'].to(device, dtype)
            curr_layer.weight.data += multiplier * diff_weight * alpha
            
        if 'diff_b' in elems:
            diff_bias = elems['diff_b'].to(device, dtype)
            if hasattr(curr_layer, 'bias') and curr_layer.bias is not None:
                curr_layer.bias.data += multiplier * diff_bias * alpha
                
        if 'diff_m' in elems:
            diff_mod = elems['diff_m'].to(device, dtype)
            # 根据具体架构，可能存在单独的 modulation 属性或直接修改 weight
            if hasattr(curr_layer, 'modulation'):
                curr_layer.modulation.data += multiplier * diff_mod * alpha
            elif hasattr(curr_layer, 'weight'):
                curr_layer.weight.data += multiplier * diff_mod * alpha

        curr_layer = curr_layer.to(origin_device, origin_dtype)

    if sequential_cpu_offload_flag:
        pipeline.enable_sequential_cpu_offload(device=offload_device)
    return pipeline


def unmerge_lora(pipeline, lora_path, multiplier=1, device="cpu", dtype=torch.float32):
    """Unmerge state_dict in LoRANetwork from the pipeline in diffusers."""
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    state_dict = load_file(lora_path)

    # 同理，保证解绑时如果不是标准格式，要先转换
    if LORA_PREFIX_UNET not in next(iter(state_dict.keys())):
        state_dict = format_lora_state_dict(state_dict)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    sequential_cpu_offload_flag = False
    if pipeline.transformer.device == torch.device(type="meta"):
        pipeline.remove_all_hooks()
        sequential_cpu_offload_flag = True

    for layer, elems in updates.items():

        if "lora_te" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.transformer

        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name + "_" + "_".join(layer_infos))
                    break
                except Exception:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name)
                        if len(layer_infos) > 0:
                            temp_name = layer_infos.pop(0)
                        elif len(layer_infos) == 0:
                            break
                    except Exception:
                        if len(layer_infos) == 0:
                            print('Error loading layer')
                        if len(temp_name) > 0:
                            temp_name += "_" + layer_infos.pop(0)
                        else:
                            temp_name = layer_infos.pop(0)

        origin_dtype = curr_layer.weight.data.dtype if hasattr(curr_layer, "weight") else curr_layer.data.dtype
        origin_device = curr_layer.weight.data.device if hasattr(curr_layer, "weight") else curr_layer.data.device

        curr_layer = curr_layer.to(device, dtype)
        
        # ---------------------------------------------------------
        # 【新增】混合卸载逻辑：支持 -= diff
        # ---------------------------------------------------------
        
        if 'lora_up.weight' in elems and 'lora_down.weight' in elems:
            weight_up = elems['lora_up.weight'].to(device, dtype)
            weight_down = elems['lora_down.weight'].to(device, dtype)
            
            if 'alpha' in elems.keys():
                alpha = elems['alpha'].item() / weight_up.shape[1]
            else:
                alpha = 1.0

            if len(weight_up.shape) == 4:
                lora_delta = torch.mm(
                    weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                lora_delta = torch.mm(weight_up, weight_down)
                
            curr_layer.weight.data -= multiplier * alpha * lora_delta

        if 'diff' in elems:
            curr_layer.weight.data -= multiplier * elems['diff'].to(device, dtype)
            
        if 'diff_b' in elems:
            if hasattr(curr_layer, 'bias') and curr_layer.bias is not None:
                curr_layer.bias.data -= multiplier * elems['diff_b'].to(device, dtype)
                
        if 'diff_m' in elems:
            diff_mod = elems['diff_m'].to(device, dtype)
            if hasattr(curr_layer, 'modulation'):
                curr_layer.modulation.data -= multiplier * diff_mod
            elif hasattr(curr_layer, 'weight'):
                curr_layer.weight.data -= multiplier * diff_mod

        curr_layer = curr_layer.to(origin_device, origin_dtype)

    if sequential_cpu_offload_flag:
        pipeline.enable_sequential_cpu_offload(device=device)
    return pipeline


@torch.no_grad()
def merge_lora_to_modules(
    transformer,
    text_encoder,
    lora_path,
    multiplier,
    device='cpu',
    dtype=torch.float32,
    state_dict=None,
    transformer_only=False
):
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if state_dict is None:
        state_dict = load_file(lora_path)
    else:
        state_dict = state_dict
        
    if LORA_PREFIX_TRANSFORMER not in next(iter(state_dict.keys())):
        state_dict = format_lora_state_dict(state_dict)
        
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    for layer, elems in updates.items():

        if LORA_PREFIX_TEXT_ENCODER in layer:
            if transformer_only:
                continue
            else:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
            curr_layer = transformer

        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name + "_" + "_".join(layer_infos))
                    break
                except Exception:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name)
                        if len(layer_infos) > 0:
                            temp_name = layer_infos.pop(0)
                        elif len(layer_infos) == 0:
                            break
                    except Exception:
                        if len(layer_infos) == 0:
                            print('Error loading layer')
                        if len(temp_name) > 0:
                            temp_name += "_" + layer_infos.pop(0)
                        else:
                            temp_name = layer_infos.pop(0)

        origin_dtype = curr_layer.weight.data.dtype if hasattr(curr_layer, "weight") else curr_layer.data.dtype
        origin_device = curr_layer.weight.data.device if hasattr(curr_layer, "weight") else curr_layer.data.device

        curr_layer = curr_layer.to(device, dtype)

        # ---------------------------------------------------------
        # 【新增】模块级别的合并同样支持混合逻辑
        # ---------------------------------------------------------
        if 'lora_up.weight' in elems and 'lora_down.weight' in elems:
            weight_up = elems['lora_up.weight'].to(device, dtype)
            weight_down = elems['lora_down.weight'].to(device, dtype)
            
            if 'alpha' in elems.keys():
                alpha = elems['alpha'].item() / weight_up.shape[1]
            else:
                alpha = 1.0

            if len(weight_up.shape) == 4:
                lora_delta = torch.mm(
                    weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                lora_delta = torch.mm(weight_up, weight_down)
                
            curr_layer.weight.data += multiplier * alpha * lora_delta

        if 'diff' in elems:
            curr_layer.weight.data += multiplier * elems['diff'].to(device, dtype)
            
        if 'diff_b' in elems:
            if hasattr(curr_layer, 'bias') and curr_layer.bias is not None:
                curr_layer.bias.data += multiplier * elems['diff_b'].to(device, dtype)
                
        if 'diff_m' in elems:
            diff_mod = elems['diff_m'].to(device, dtype)
            if hasattr(curr_layer, 'modulation'):
                curr_layer.modulation.data += multiplier * diff_mod
            elif hasattr(curr_layer, 'weight'):
                curr_layer.weight.data += multiplier * diff_mod

        curr_layer = curr_layer.to(origin_device, origin_dtype)

    return transformer, text_encoder




