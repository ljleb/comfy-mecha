import functools
import gc
import pathlib
import re
from typing import List, Tuple, Iterable, Any, Optional
import sd_mecha
import torch.cuda
import tqdm
from sd_mecha.extensions import merge_methods, merge_spaces
from sd_mecha.extensions.merge_methods import MergeMethod
from sd_mecha.recipe_merger import LoadInputDictsVisitor, CloseInputDictsVisitor, open_input_dicts
import folder_paths
import comfy
from comfy import model_management
from comfy.sd import load_state_dict_guess_config
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
    for k, v in prompt_executor.caches.outputs.cache.copy().items():
        for v in v:
            for v in v:
                if v in temporary_merged_objects and k in prompt_executor.caches.outputs.cache:
                    prompt_executor.caches.outputs.cache.pop(k)

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
        with open_input_dicts(
            recipe,
            [pathlib.Path(p) for p in
             folder_paths.get_folder_paths("checkpoints") + folder_paths.get_folder_paths("loras")],
            buffer_size_per_dict=0,
        ):
            return sd_mecha.serialize(recipe),


class MechaDeserializer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe_txt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                }),
            },
        }

    RETURN_TYPES = "MECHA_RECIPE",
    RETURN_NAMES = "recipe",
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(self, recipe_txt: str):
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
                "default_merge_device": (["none"] + all_torch_devices, {
                    "default": main_torch_device,
                }),
                "default_merge_dtype": (["none"] + list(DTYPE_MAPPING.keys()), {
                    "default": "fp64",
                }),
                "output_device": (["none"] + all_torch_devices, {
                    "default": "cpu",
                }),
                "output_dtype": (["none"] + list(DTYPE_MAPPING.keys()), {
                    "default": "bf16",
                }),
                "total_buffer_size": ("STRING", {
                    "default": "0.5G",
                }),
                "threads": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 16,
                    "step": 1,
                }),
                "temporary_merge": ("BOOLEAN", {
                    "default": True,
                }),
                "strict_weight_space": ("BOOLEAN", {
                    "default": True,
                })
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "recipe_txt")
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
        strict_weight_space: bool,
    ):
        global temporary_merged_recipes, prompt_executor
        total_buffer_size = memory_to_bytes(total_buffer_size)

        with open_input_dicts(
            recipe,
            [pathlib.Path(p) for p in folder_paths.get_folder_paths("checkpoints") + folder_paths.get_folder_paths("loras")],
            buffer_size_per_dict=0,
        ):
            recipe_txt = sd_mecha.serialize(recipe)

        try:
            already_merged_index = [t[0] for t in temporary_merged_recipes].index(recipe_txt)
            return temporary_merged_recipes[already_merged_index][1]
        except ValueError:
            if temporary_merged_recipes:
                free_temporary_merges(prompt_executor)

        model_config = getattr(recipe.model_config, "identifier", None)
        if fallback_model == "none" or not model_config:
            fallback_model = None
        else:
            fallback_model = sd_mecha.model(fallback_model, model_config=model_config)

        merger = sd_mecha.RecipeMerger(
            models_dir=folder_paths.get_folder_paths("checkpoints") + folder_paths.get_folder_paths("loras"),
            default_device=default_merge_device if default_merge_device != "none" else None,
            default_dtype=DTYPE_MAPPING[default_merge_dtype] if default_merge_dtype != "none" else None,
            tqdm=ComfyTqdm,
        )
        state_dict = {}
        model_management.unload_all_models()
        merger.merge_and_save(
            recipe=recipe,
            output=state_dict,
            fallback_model=fallback_model,
            save_device=output_device if output_device != "none" else None,
            save_dtype=DTYPE_MAPPING[output_dtype] if output_dtype != "none" else None,
            threads=threads if threads >= 0 else None,
            total_buffer_size=total_buffer_size,
            strict_weight_space=strict_weight_space,
        )
        res = load_state_dict_guess_config(state_dict, embedding_directory=folder_paths.get_folder_paths("embeddings"))[:3]
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


class MechaModelRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    [
                        f
                        for f in folder_paths.get_filename_list("checkpoints") + folder_paths.get_filename_list("loras")
                        if f.endswith(".safetensors")
                    ],
                ),
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
    ):
        return sd_mecha.model(model_path),


class MechaLoraRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ([f for f in folder_paths.get_filename_list("loras") if f.endswith(".safetensors")],),
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
    ):
        recipe = sd_mecha.model(model_path)
        try:
            recipe.accept(LoadInputDictsVisitor(
                [pathlib.Path(p) for p in folder_paths.get_folder_paths("loras")],
                0,
            ))

            if recipe.model_config.identifier == "sdxl-kohya_kohya_lora":
                recipe = sd_mecha.convert(recipe, "sdxl-sgm")
            elif recipe.model_config.identifier == "sd1-kohya_kohya_lora":
                recipe = sd_mecha.convert(recipe, "sd1-ldm")
            else:
                raise RuntimeError(f"unsupported lora model config: {recipe.model_config.identifier}")
        finally:
            recipe.accept(CloseInputDictsVisitor())

        return recipe,


def register_merge_methods():
    for method in merge_methods.get_all():
        method_name = method.get_identifier()
        class_name = f"{snake_case_to_upper(method_name)}MechaRecipe"
        short_title_name = snake_case_to_title(method_name)
        title_name = f"{snake_case_to_title(method_name)} Mecha Recipe"
        NODE_CLASS_MAPPINGS[title_name] = make_comfy_node_class(class_name, method)
        NODE_DISPLAY_NAME_MAPPINGS[title_name] = short_title_name


def make_comfy_node_class(class_name: str, method: MergeMethod) -> type:
    param_names = method.get_param_names()
    input_merge_spaces = method.get_input_merge_spaces()
    merge_spaces_dict = input_merge_spaces.as_dict(0)
    default_args = method.get_default_args()
    len_mandatory_args = len(param_names.args) - len(default_args.args)

    return type(class_name, (object,), {
        "INPUT_TYPES": lambda: {
            "required": {
                **{
                    f"{name} ({'|'.join(sorted(merge_spaces.get_identifiers(merge_spaces_dict[index])))})": ("MECHA_RECIPE",)
                    for index, name in enumerate(param_names.args[:len_mandatory_args])
                },
                **{
                    f"{name} ({'|'.join(sorted(merge_spaces.get_identifiers(merge_spaces_dict[index])))})": ("MECHA_RECIPE",)
                    for index, name in sorted(param_names.kwargs.items(), key=lambda t: t[0])
                    if index not in default_args.kwargs
                },
            },
            "optional": {
                **{
                    f"{name} ({default_args.args[default_index]})": ("MECHA_RECIPE", {"default": default_args.args[default_index]})
                    for default_index, name in enumerate(param_names.args[len_mandatory_args:])
                },
                **{
                    f"{name} ({default_args.kwargs[index]})": ("MECHA_RECIPE", {"default": default_args.kwargs[index]})
                    for index, name in sorted(param_names.kwargs.items(), key=lambda t: t[0])
                    if index in default_args.kwargs
                },
                **({
                    f"{param_names.vararg} ({'|'.join(sorted(merge_spaces.get_identifiers(input_merge_spaces.vararg)))})": ("MECHA_RECIPE_LIST", {"default": []}),
                } if param_names.has_varargs() else {}),
                "_use_cache": ("BOOLEAN", {
                    "default": False,
                }),
            }
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
    param_names = method.get_param_names()

    def execute(*_args, **kwargs):
        # remove default values / merge space from keys
        # comfy nodes cannot distinguish display names from id names
        # in consequence we have to unmangle things here
        for k in list(kwargs):
            if " (" in k and k.endswith(")"):
                new_k = k.split(" ")[0]
                kwargs[new_k] = kwargs[k]
                del kwargs[k]

        use_cache = kwargs.pop("_use_cache")

        args = [kwargs[m] for m in param_names.args]
        if param_names.has_varargs():
            args.extend(kwargs[param_names.vararg])
        kwargs = {
            k: kwargs[k]
            for k in param_names.kwargs
            if k in kwargs
        }
        recipe = method(*args, **kwargs)
        if method.identifier == "add_difference":
            recipe = recipe | args[0]

        if use_cache:
            recipe.set_cache()
        return recipe,

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
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


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
    "Model Mecha Recipe": "Model",
    "Lora Mecha Recipe": "Lora",
    "Mecha Recipe List": "Recipe List",
}

register_merge_methods()
