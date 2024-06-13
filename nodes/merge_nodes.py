import functools
import gc
import logging
import pathlib
import re
import textwrap
from typing import TypeVar, Optional, List

import sd_mecha
import torch.cuda
import tqdm
from sd_mecha import Hyper
from sd_mecha.extensions.merge_method import MergeMethod, convert_to_recipe
from sd_mecha.merge_methods import SameMergeSpace, LiftFlag, MergeSpace
from torch import Tensor
import folder_paths
import comfy
from comfy import model_management, model_detection
from comfy.sd import CLIP
import execution


cached_merges_to_delete = list()
prompt_executor: Optional[execution.PromptExecutor] = None


def patch_prompt_executor():
    patch_key = "__mecha_execute_original"
    if not hasattr(execution.PromptExecutor, patch_key):
        setattr(execution.PromptExecutor, patch_key, execution.PromptExecutor.execute)
        execution.PromptExecutor.execute = functools.partialmethod(prompt_executor_execute, __original_function=execution.PromptExecutor.execute)


def prompt_executor_execute(self, *args, __original_function, **kwargs):
    global prompt_executor
    prompt_executor = self
    free_cached_merges(self)
    return __original_function(self, *args, **kwargs)


def free_cached_merges(prompt_executor: execution.PromptExecutor):
    global cached_merges_to_delete
    if not cached_merges_to_delete:
        return

    for k, v in prompt_executor.outputs.copy().items():
        for v in v:
            for v in v:
                if v in cached_merges_to_delete and k in prompt_executor.outputs:
                    prompt_executor.outputs.pop(k)

    del k, v
    cached_merges_to_delete.clear()
    model_management.cleanup_models()
    gc.collect()
    model_management.soft_empty_cache()


patch_prompt_executor()


class MechaMerger:
    @classmethod
    def INPUT_TYPES(cls):
        all_torch_devices = get_all_torch_devices()
        main_torch_device = model_management.get_torch_device().type
        return {
            "required": {
                "recipe": ("MECHA_RECIPE",),
                "fallback_model": (["none"] + [f for f in folder_paths.get_filename_list("checkpoints") if f.endswith(".safetensors")], {
                    "default": "none",
                }),
                "default_merge_device": (all_torch_devices, {
                    "default": main_torch_device,
                }),
                "default_merge_dtype": (list(DTYPE_MAPPING.keys()), {
                    "default": "fp64",
                }),
                "output_device": (all_torch_devices, {
                    "default": "cpu",
                }),
                "output_dtype": (list(DTYPE_MAPPING.keys()), {
                    "default": "fp16",
                }),
                "total_buffer_size": ("STRING", {
                    "default": "0.5G",
                }),
                "threads": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                }),
                "temporary_merge": (["True", "False"], {
                    "default": "True",
                })
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
        total_buffer_size: str,
        threads: int,
        temporary_merge: str,
    ):
        global cached_merges_to_delete, prompt_executor
        temporary_merge = temporary_merge == "True"
        total_buffer_size = memory_to_bytes(total_buffer_size)

        model_management.unload_all_models()
        free_cached_merges(prompt_executor)

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
        res = load_checkpoint_guess_config(state_dict)
        if temporary_merge:
            cached_merges_to_delete.extend(res)
        return res


memory_units = ('B', 'K', 'M', 'G')
memory_suffix_re = re.compile(rf'(\d+(\.\d+)?)\s*([{"".join(memory_units)}]?)$')
memory_units_map = {u: 1024 ** i for i, u in enumerate(memory_units)}


def memory_to_bytes(memory_str: str) -> int:
    """
    Convert a memory size string containing multiple terms (like '1G 500K') to the total number of bytes.

    Args:
    memory_str (str): Memory size string with multiple terms, each consisting of a number followed by a unit:
                      B - bytes
                      K - kilobytes
                      M - megabytes
                      G - gigabytes
                      T - terabytes
                      P - petabytes

    Returns:
    int: The total equivalent memory in bytes, always as an integer.
    """
    total_bytes = 0
    terms = memory_str.upper().split()
    for term in terms:
        match = memory_suffix_re.match(term)
        if not match:
            continue

        number = float(match.group(1))
        unit = match.group(3) if match.group(3) else 'B'
        total_bytes += number * memory_units_map[unit]

    return round(total_bytes)


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

    clip_target = model_config.clip_target(state_dict=state_dict)
    if clip_target is not None:
        clip_sd = model_config.process_clip_state_dict(state_dict)
        if len(clip_sd) > 0:
            clip = CLIP(clip_target, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            m, u = clip.load_sd(clip_sd, full_model=True)
            if len(m) > 0:
                m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                if len(m_filter) > 0:
                    logging.warning("clip missing: {}".format(m))
                else:
                    logging.debug("clip missing: {}".format(m))

            if len(u) > 0:
                logging.debug("clip unexpected {}:".format(u))
        else:
            logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

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
        logging.info("loaded straight to GPU")
        model_management.load_model_gpu(model_patcher)

    return model_patcher, clip


class MechaModelRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ([f for f in folder_paths.get_filename_list("checkpoints") if f.endswith(".safetensors")],),
                "model_arch": (sd_mecha.extensions.model_arch.get_all(),),
            },
        }
    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        model_path: str,
        model_arch: str,
    ):
        for path in folder_paths.get_folder_paths("checkpoints"):
            model_path_candidate = pathlib.Path(path, model_path)
            if model_path_candidate.exists():
                model_path = model_path_candidate
                break

        return sd_mecha.model(model_path, model_arch=model_arch),


class MechaLoraRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ([f for f in folder_paths.get_filename_list("loras") if f.endswith(".safetensors")],),
                "model_arch": (sd_mecha.extensions.model_arch.get_all(),),
            },
        }
    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        model_path: str,
        model_arch: str,
    ):
        for path in folder_paths.get_folder_paths("loras"):
            model_path_candidate = pathlib.Path(path, model_path)
            if model_path_candidate.exists():
                model_path = model_path_candidate
                break

        return sd_mecha.lora(model_path, model_arch=model_arch),


def register_merge_methods():
    for method_name in sd_mecha.extensions.merge_method._merge_methods_registry:
        method = sd_mecha.extensions.merge_method.resolve(method_name)

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
                    hyper_name: ("MECHA_HYPER",)
                    for hyper_name in method.get_hyper_names() - method.get_volatile_hyper_names()
                    if hyper_name not in method.get_default_hypers()
                },
                "device": (["default", *get_all_torch_devices()], {
                    "default": "default",
                }),
                "dtype": (list(OPTIONAL_DTYPE_MAPPING.keys()), {
                    "default": "default",
                }),
            },
            "optional": {
                **({
                    method.get_model_varargs_name(): ("MECHA_RECIPE_LIST", {"default": []}),
                } if method.get_model_varargs_name() is not None else {}),
                **{
                    hyper_name: ("MECHA_HYPER", {"default": method.get_default_hypers()[hyper_name]})
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


MAX_VARARGS_MODELS = 64  # arbitrary limit to n-models methods (open an issue if this is a problem)


class MechaRecipeList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 2, "min": 0, "max": MAX_VARARGS_MODELS, "step": 1})
            },
            "optional": {
                f"recipe_{i}": ("MECHA_RECIPE",)
                for i in range(MAX_VARARGS_MODELS)
            }
        }
    RETURN_TYPES = ("MECHA_RECIPE_LIST",)
    RETURN_NAMES = ("recipes",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        count: int,
        **kwargs,
    ):
        return [kwargs[f"recipe_{i}"] for i in range(count)],


def get_all_torch_devices() -> List[str]:
    torch_device = model_management.get_torch_device().type
    return [
        "cpu",
        *([torch_device] if torch_device != "cpu" else []),
        *([f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []),
    ]


def get_method_node_execute(method: MergeMethod):
    def execute(*_args, **kwargs):
        dtype = OPTIONAL_DTYPE_MAPPING[kwargs["dtype"]]
        device = kwargs["device"]
        if device == "default":
            device = None

        models = [kwargs[m] for m in method.get_model_names()]
        if method.get_model_varargs_name() is not None:
            models.extend(kwargs[method.get_model_varargs_name()])

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


class MechaCustomCodeRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        all_cuda_devices = ["cpu", *(["cuda", *[f"cuda:{i}" for i in range(torch.cuda.device_count())]] if torch.cuda.is_available() else [])]
        return {
            "required": {
                "device": (["default", *all_cuda_devices], {
                    "default": "default",
                }),
                "dtype": (list(OPTIONAL_DTYPE_MAPPING.keys()), {
                    "default": "default",
                }),
                "script": ("STRING", {
                    "default": default_custom_code_method,
                    "multiline": True,
                }),
            },
            "optional": {
                **{
                    k: ("MECHA_RECIPE", {
                        "default": "none",
                    })
                    for k in ["a", "b", "c", "d", "e"]
                },
                **{
                    hyper_name: ("*", {"default": 0.0})
                    for hyper_name in ["alpha", "beta", "gamma", "omega", "sigma"]
                },
            },
        }
    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        script: str,
        device: str,
        dtype: str,
        **kwargs,
    ):
        if device == "default":
            device = None

        script_locals = {}
        script_globals = {
            torch.__name__: torch,
            Tensor.__name__: Tensor,
            TypeVar.__name__: TypeVar,
            LiftFlag.__name__: LiftFlag,
            MergeSpace.__name__: MergeSpace,
            "Hyper": Hyper,
        }
        exec(script, script_globals, script_locals)
        method: MergeMethod = convert_to_recipe(script_locals["main"], register=False).__wrapped_method__
        models = [
            m for k, m in kwargs.items()
            if k in method.get_model_names() and m != "none"
        ]
        hypers = {
            k: m for k, m in kwargs.items()
            if k in method.get_hyper_names()
        }
        return method.create_recipe(*models, **hypers, device=device, dtype=OPTIONAL_DTYPE_MAPPING[dtype]),

    @classmethod
    def IS_CHANGED(cls, script: str, **kwargs):
        return script


default_custom_code_method = textwrap.dedent(f"""
BaseSpace = LiftFlag[{MergeSpace.__name__}.BASE]
DeltaSpace = LiftFlag[{MergeSpace.__name__}.DELTA]
SameMergeSpace = {TypeVar.__name__}("SameMergeSpace", bound=LiftFlag[MergeSpace.BASE | MergeSpace.DELTA])

def main(
    a: {Tensor.__name__} | SameMergeSpace,
    b: {Tensor.__name__} | SameMergeSpace,
#    c: {Tensor.__name__} | SameMergeSpace,
#    d: {Tensor.__name__} | SameMergeSpace,
#    e: {Tensor.__name__} | SameMergeSpace,
    *,
    alpha: Hyper = 0.0,
    beta: Hyper = 0.0,
    gamma: Hyper = 0.0,
    omega: Hyper = 0.0,
    sigma: Hyper = 0.0,
    **kwargs,
) -> {Tensor.__name__} | {SameMergeSpace.__name__}:
    return (1 - alpha) * a + alpha * b
""")


DTYPE_MAPPING = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}
OPTIONAL_DTYPE_MAPPING = {
    "default": None,
} | DTYPE_MAPPING


NODE_CLASS_MAPPINGS = {
    "Mecha Merger": MechaMerger,
    "Model Mecha Recipe": MechaModelRecipe,
    "Lora Mecha Recipe": MechaLoraRecipe,
    "Mecha Recipe List": MechaRecipeList,
    "Mecha Custom Code Recipe": MechaCustomCodeRecipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mecha Merger": "Merger",
    "Mecha Model Recipe": "Model",
    "Lora Mecha Recipe": "Lora",
    "Mecha Recipe List": "Recipe List",
    "Mecha Custom Code Recipe": "Custom Code",
}

register_merge_methods()
