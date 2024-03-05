import sd_mecha
import torch.cuda
import tqdm
from sd_mecha.extensions.merge_method import MergeMethod

import folder_paths
import comfy
from comfy import model_management, model_detection
from comfy.sd import CLIP


class MechaMerger:
    @classmethod
    def INPUT_TYPES(cls):
        all_cuda_devices = ["cpu", *(["cuda", *[f"cuda:{i}" for i in range(torch.cuda.device_count())]] if torch.cuda.is_available() else [])]
        return {
            "required": {
                "recipe": ("MECHA_RECIPE",),
                "fallback_model": (["none"] + [f for f in folder_paths.get_filename_list("checkpoints") if f.endswith(".safetensors")], {
                    "default": "none",
                }),
                "default_merge_device": (all_cuda_devices, {
                    "default": "cuda" if torch.cuda.is_available() else "cpu",
                }),
                "default_merge_dtype": (list(DTYPE_MAPPING.keys()), {
                    "default": "fp64",
                }),
                "output_device": (all_cuda_devices, {
                    "default": "cpu",
                }),
                "output_dtype": (list(DTYPE_MAPPING.keys()), {
                    "default": "fp16",
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
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        recipe: sd_mecha.recipe_nodes.RecipeNode,
        fallback_model: str,
        output_device: str,
        output_dtype: str,
        default_merge_device: str,
        default_merge_dtype: str,
        total_buffer_size: int,
        threads: int,
    ):
        model_arch = getattr(recipe.model_arch, "identifier", None)
        if fallback_model == "none" or not model_arch:
            fallback_model = None
        else:
            fallback_model = sd_mecha.model(fallback_model, model_arch=model_arch)

        merger = sd_mecha.RecipeMerger(
            models_dir=folder_paths.get_folder_paths("checkpoints"),
            default_device=default_merge_device,
            default_dtype=DTYPE_MAPPING[default_merge_dtype],
            tqdm=ComfyTqdm,
        )
        state_dict = {}
        model_management.unload_all_models()
        merger.merge_and_save(
            recipe=recipe,
            output=state_dict,
            fallback_model=fallback_model,
            save_dtype=DTYPE_MAPPING[output_dtype],
            save_device=output_device,
            threads=threads if threads > 0 else None,
            total_buffer_size=total_buffer_size,
        )
        return load_checkpoint_guess_config(state_dict)


class ComfyTqdm:
    def __init__(self, *args, **kwargs):
        self.progress = tqdm.tqdm(*args, **kwargs)
        self.comfy_progress = comfy.utils.ProgressBar(kwargs["total"])

    def update(self):
        self.progress.update()
        self.comfy_progress.update_absolute(self.progress.n, self.progress.total)

    def __getattr__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            return getattr(self.progress, item)


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


class ModelMechaRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ([f for f in folder_paths.get_filename_list("checkpoints") if f.endswith(".safetensors")],),
                "model_arch": (sd_mecha.extensions.model_arch.get_all(),),
                "model_type": (["base", "lora"], {"default": "base"}),
            },
        }
    RETURN_TYPES = ("MECHA_RECIPE",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        model_path: str,
        model_arch: str,
        model_type: str,
    ):
        return sd_mecha.model(model_path, model_arch=model_arch, model_type=model_type),


NODE_CLASS_MAPPINGS = {
    "Mecha Merger": MechaMerger,
    "Model Mecha Recipe": ModelMechaRecipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mecha Merger": "Merger",
    "Model Mecha Recipe": "Model",
}


def register_methods():
    for method_name in sd_mecha.extensions.merge_method._merge_methods_registry:
        method = sd_mecha.extensions.merge_method.resolve(method_name)
        if method.get_model_varargs_name() is not None:
            continue

        class_name = f"{snake_case_to_upper(method_name)}MechaRecipe"
        short_title_name = snake_case_to_title(method_name)
        title_name = f"{snake_case_to_title(method_name)} Mecha Recipe"
        NODE_CLASS_MAPPINGS[title_name] = make_comfy_node_class(class_name, method)
        NODE_DISPLAY_NAME_MAPPINGS[title_name] = short_title_name


def make_comfy_node_class(class_name: str, method: MergeMethod) -> type:
    return type(class_name, (object,), {
        "INPUT_TYPES": lambda: {
            "required": {
                **{
                    model_name: ("MECHA_RECIPE",)
                    for model_name, merge_space in zip(method.get_model_names(), method.get_input_merge_spaces()[0])
                },
                **{
                    hyper_name: ("*",)
                    for hyper_name in method.get_hyper_names() - method.get_volatile_hyper_names()
                    if hyper_name not in method.get_default_hypers()
                },
                "device": (["default", "cpu", *(["cuda", *[f"cuda:{i}" for i in range(torch.cuda.device_count())]] if torch.cuda.is_available() else [])], {
                    "default": "default",
                }),
                "dtype": (list(OPTIONAL_DTYPE_MAPPING.keys()), {
                    "default": "default",
                }),
            },
            "optional": {
                **{
                    hyper_name: ("*", {"default": method.get_default_hypers()[hyper_name]})
                    for hyper_name in method.get_hyper_names() - method.get_volatile_hyper_names()
                    if hyper_name in method.get_default_hypers()
                },
            },
        },
        "RETURN_TYPES": ("MECHA_RECIPE",),
        "RETURN_NAMES": ("recipe",),
        "FUNCTION": "execute",
        "OUTPUT_NODE": False,
        "CATEGORY": "advanced/model_merging/mecha",
        "execute": get_method_node_execute(method),
    })


def get_method_node_execute(method: MergeMethod):
    def execute(*_args, **kwargs):
        dtype = OPTIONAL_DTYPE_MAPPING[kwargs["dtype"]]
        device = kwargs["device"]
        if device == "default":
            device = None

        models = [kwargs[m] for m in method.get_model_names()]
        hypers = {
            k: kwargs[k]
            for k in method.get_hyper_names()
            if k in kwargs
        }

        return method.create_recipe(*models, **hypers, dtype=dtype, device=device),

    return execute


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


register_methods()
