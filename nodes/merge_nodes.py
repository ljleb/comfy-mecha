import functools
import gc
import logging
import re
from typing import Optional, List, Tuple, Any, Iterable
import sd_mecha
import torch.cuda
import tqdm
from sd_mecha.extensions.merge_method import MergeMethod
import folder_paths
import comfy
from comfy import model_management, model_detection
from comfy.sd import CLIP
import execution


temporary_merged_recipes: List[Tuple[str, Iterable[Any]]] = []
cached_mergers_count: int = 0
temporary_mergers_count: int = 0
prompt_executor: Optional[execution.PromptExecutor] = None


def patch_prompt_executor():
    patch_key = "__mecha_execute_original"
    if not hasattr(execution.PromptExecutor, patch_key):
        setattr(execution.PromptExecutor, patch_key, execution.PromptExecutor.execute)
        execution.PromptExecutor.execute = functools.partialmethod(prompt_executor_execute, __original_function=execution.PromptExecutor.execute)


def prompt_executor_execute(self, *args, __original_function, **kwargs):
    global cached_mergers_count, temporary_mergers_count, prompt_executor
    prompt_executor = self
    cached_mergers_count = 0
    temporary_mergers_count = 0
    try:
        return __original_function(self, *args, **kwargs)
    finally:
        if cached_mergers_count > 0 and temporary_mergers_count > 0:
            free_temporary_merges(self)


def free_temporary_merges(prompt_executor: execution.PromptExecutor):
    global temporary_merged_recipes
    if not temporary_merged_recipes:
        return

    temporary_merged_objects = [e for t in temporary_merged_recipes for e in t[1]]
    for k, v in prompt_executor.outputs.copy().items():
        for v in v:
            for v in v:
                if v in temporary_merged_objects and k in prompt_executor.outputs:
                    prompt_executor.outputs.pop(k)

    del k, v
    temporary_merged_recipes.clear()
    model_management.cleanup_models()
    gc.collect()
    model_management.soft_empty_cache()


patch_prompt_executor()


class MechaSerializer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe": ("MECHA_RECIPE",),
            },
        }

    RETURN_TYPES = "STRING",
    RETURN_NAMES = "recipe_txt",
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(self, recipe):
        return sd_mecha.serialize(recipe),


class MechaDeserializer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe_txt": ("STRING", {
                    "default": "",
                    "forceInput": True,
                }),
            },
        }

    RETURN_TYPES = "MECHA_RECIPE",
    RETURN_NAMES = "recipe",
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(self, recipe_txt):
        return sd_mecha.deserialize(recipe_txt.split("\n")),


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
                "temporary_merge": ("BOOLEAN", {
                    "default": True,
                })
            },
        }
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "recipe_txt")
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    @classmethod
    def IS_CHANGED(cls, temporary_merge, **_kwargs):
        global cached_mergers_count, temporary_mergers_count
        cached_mergers_count += int(not temporary_merge)
        temporary_mergers_count += int(temporary_merge)
        return ""

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
        temporary_merge: bool,
    ):
        global temporary_merged_recipes, prompt_executor
        total_buffer_size = memory_to_bytes(total_buffer_size)

        recipe_txt = sd_mecha.serialize(recipe)
        try:
            already_merged_index = [t[0] for t in temporary_merged_recipes].index(recipe_txt)
            already_merged_recipe = temporary_merged_recipes[already_merged_index]
        except ValueError:
            already_merged_recipe = None
        if already_merged_recipe is not None:
            return already_merged_recipe[1]
        if temporary_merged_recipes and already_merged_recipe is None:
            free_temporary_merges(prompt_executor)

        model_arch = getattr(recipe.model_arch, "identifier", None)
        if fallback_model == "none" or not model_arch:
            fallback_model = None
        else:
            fallback_model = sd_mecha.model(fallback_model, model_arch=model_arch)

        merger = sd_mecha.RecipeMerger(
            models_dir=folder_paths.get_folder_paths("checkpoints") + folder_paths.get_folder_paths("loras"),
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
            temporary_merged_recipes.append((recipe_txt, res))
        return *res, recipe_txt


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
    all_hyper_names = method.get_hyper_names() - method.get_volatile_hyper_names()
    return type(class_name, (object,), {
        "INPUT_TYPES": lambda: {
            "required": {
                **{
                    f"{model_name} ({merge_space})": ("MECHA_RECIPE",)
                    for model_name, merge_space in zip(method.get_model_names(), method.get_input_merge_spaces()[0])
                },
                **{
                    hyper_name: ("MECHA_HYPER",)
                    for hyper_name in all_hyper_names
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
                    f"{method.get_model_varargs_name()} ({method.get_input_merge_spaces()[1]})": ("MECHA_RECIPE_LIST", {"default": []}),
                } if method.get_model_varargs_name() is not None else {}),
                **{
                    f"{hyper_name} ({method.get_default_hypers()[hyper_name]})": ("MECHA_HYPER", {"default": method.get_default_hypers()[hyper_name]})
                    for hyper_name in all_hyper_names
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
        return [kwargs[f"recipe_{i}"] for i in range(count) if f"recipe_{i}" in kwargs],


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

        # remove default values / merge space from keys
        # comfy nodes cannot distinguish display names from id names
        # in consequence we have to unmangle things here
        for k in list(kwargs):
            if " (" in k and k.endswith(")"):
                new_k = k.split(" ")[0]
                kwargs[new_k] = kwargs[k]
                del kwargs[k]

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
    "Mecha Serializer": MechaSerializer,
    "Mecha Deserializer": MechaDeserializer,
    "Model Mecha Recipe": MechaModelRecipe,
    "Lora Mecha Recipe": MechaLoraRecipe,
    "Mecha Recipe List": MechaRecipeList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mecha Merger": "Merger",
    "Mecha Serializer": "Serializer",
    "Mecha Deserializer": "Deserializer",
    "Mecha Model Recipe": "Model",
    "Lora Mecha Recipe": "Lora",
    "Mecha Recipe List": "Recipe List",
}

register_merge_methods()
