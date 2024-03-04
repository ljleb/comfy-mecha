from typing import Optional
import sd_mecha
import torch.cuda
from sd_mecha.extensions.merge_method import MergeMethod
import folder_paths
import comfy
from comfy import model_management, model_detection
from comfy.sd import CLIP


def snake_case_to_upper(name: str):
    i = 0
    while i < len(name):
        if name[i] == "_":
            name = name[:i] + name[i+1:i+2].upper() + name[i+2:]
        i += 1

    return name[:1].upper() + name[1:]


def snake_case_to_title(name: str):
    i = 0
    while i < len(name):
        if name[i] == "_":
            name = name[:i] + " " + name[i+1:i+2].upper() + name[i+2:]
        i += 1

    return name[:1].upper() + name[1:]


DTYPE_MAPPING = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}

OPTIONAL_DTYPE_MAPPING = {
    "default": None,
} | DTYPE_MAPPING


class MechaMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe": ("MECHA_RECIPE",),
                "device": (["cpu", *(["cuda", *[f"cuda:{i}" for i in range(torch.cuda.device_count())]] if torch.cuda.is_available() else [])], {
                    "default": "cpu",
                }),
                "dtype": (list(DTYPE_MAPPING.keys()), {
                    "default": "fp16",
                }),
                "default_merge_device": (["cpu", *(["cuda", *[f"cuda:{i}" for i in range(torch.cuda.device_count())]] if torch.cuda.is_available() else [])], {
                    "default": "cuda" if torch.cuda.is_available() else "cpu",
                }),
                "default_merge_dtype": (list(DTYPE_MAPPING.keys()), {
                    "default": "fp64",
                }),
                "total_buffer_size": ("INT", {
                    "default": 2**28,
                    "min": 2**8,
                    "max": 2**34,
                    "step": 2**8,
                }),
                "threads": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                }),
            },
            "optional": {
                "fallback_model": ("MECHA_RECIPE", {
                    "default": None,
                }),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        recipe: sd_mecha.recipe_nodes.RecipeNode,
        device: str,
        dtype: str,
        default_merge_device: str,
        default_merge_dtype: str,
        total_buffer_size: int,
        fallback_model: Optional[sd_mecha.recipe_nodes.RecipeNode],
        threads: int,
    ):
        merger = sd_mecha.RecipeMerger(
            models_dir=folder_paths.models_dir,
            default_device=default_merge_device,
            default_dtype=DTYPE_MAPPING[default_merge_dtype],
        )
        state_dict = {}
        merger.merge_and_save(
            recipe=recipe,
            output=state_dict,
            fallback_model=fallback_model,
            save_dtype=DTYPE_MAPPING[dtype],
            save_device=device,
            threads=threads if threads > 0 else None,
            total_buffer_size=total_buffer_size,
        )
        return load_checkpoint_guess_config(state_dict)


def load_checkpoint_guess_config(state_dict):
    clip = None

    # this code is going to rot like crazy because of hard coded key prefixes
    # I'm not going to bother with it because comfyui has this hardcoded too
    parameters = comfy.utils.calculate_parameters(state_dict, "model.diffusion_model.")
    load_device = model_management.get_torch_device()

    model_config = model_detection.model_config_from_unet(state_dict, "model.diffusion_model.")
    unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
    model = model_config.get_model(state_dict, "model.diffusion_model.", device=inital_load_device)
    model.load_model_weights(state_dict, "model.diffusion_model.")

    clip_target = model_config.clip_target()
    if clip_target is not None:
        clip_sd = model_config.process_clip_state_dict(state_dict)
        if len(clip_sd) > 0:
            clip = CLIP(clip_target)
            m, u = clip.load_sd(clip_sd, full_model=True)
            if len(m) > 0:
                print("clip missing:", m)

            if len(u) > 0:
                print("clip unexpected:", u)
        else:
            print("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    # skip vae
    left_over = [k for k in state_dict.keys() if not k.startswith("first_stage_model.")]
    if len(left_over) > 0:
        print("left over keys:", left_over)

    model_patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=load_device,
        offload_device=model_management.unet_offload_device(),
        current_device=inital_load_device,
    )
    if inital_load_device != torch.device("cpu"):
        print("loaded straight to GPU")
        model_management.load_model_gpu(model_patcher)

    return model_patcher, clip


NODE_CLASS_MAPPINGS = {
    "Mecha Merger": MechaMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mecha Merger": "Mecha Merger",
}


def register_methods():
    for method_name in sd_mecha.extensions.merge_method._merge_methods_registry:
        method = sd_mecha.extensions.merge_method.resolve(method_name)
        if method.get_model_varargs_name() is not None:
            continue

        class_name = f"{snake_case_to_upper(method_name)}MechaMethod"
        short_title_name = snake_case_to_title(method_name)
        title_name = f"{snake_case_to_title(method_name)} Mecha Method"
        NODE_CLASS_MAPPINGS[title_name] = make_comfy_node_class(class_name, method)
        NODE_DISPLAY_NAME_MAPPINGS[title_name] = short_title_name


def make_comfy_node_class(class_name: str, method: MergeMethod) -> type:
    return type(class_name, (object,), {
        "INPUT_TYPES": lambda: {
            "required": {
                **{
                    model_name: ("MECHA_RECIPE",)
                    for model_name in method.get_model_names()
                },
                **{
                    hyper_name: ("HYPER", {"default": 0.0})
                    for hyper_name in method.get_hyper_names()
                },
                "device": (["default", "cpu", *(["cuda", *[f"cuda:{i}" for i in range(torch.cuda.device_count())]] if torch.cuda.is_available() else [])], {
                    "default": "default",
                }),
                "dtype": (list(OPTIONAL_DTYPE_MAPPING.keys()), {
                    "default": "default",
                }),
            },
        },
        "RETURN_TYPES": ("MECHA_RECIPE",),
        "FUNCTION": "execute",
        "OUTPUT_NODE": False,
        "CATEGORY": "advanced/model_merging/mecha",
        "execute": lambda **kwargs: method(*[m for m in method.get_model_names()], **get_method_kwargs(method, kwargs))
    })


def get_method_kwargs(method, kwargs):
    kwargs["dtype"] = OPTIONAL_DTYPE_MAPPING[kwargs["dtype"]]
    if kwargs["device"] == "default":
        kwargs["device"] = None
    return {
        k: kwargs[k]
        for k in method.get_hyper_names()
    }


register_methods()
