import dataclasses
import operator
import comfy
import execution
import folder_paths
import functools
import gc
import pathlib
import re
import sd_mecha
import torch.cuda
import tqdm
from sd_mecha import open_graph
from sd_mecha.extensions import merge_methods, merge_spaces
from sd_mecha.recipe_nodes import MergeRecipeNode, ModelRecipeNode, LiteralRecipeNode, RecipeNode
from comfy import model_management
from comfy.model_patcher import LazyCastingParam
from comfy.sd import load_state_dict_guess_config
from typing import List, Tuple, Iterable, Any, Optional
from . import cache_nodes
from .throttling import RunLastAtMostEvery
from .transport import ComfyMechaRecipe


merge_checkpointing = sd_mecha.extensions.merge_methods.resolve("merge_checkpointing")
ALL_CONVERTERS = [m.identifier for m in merge_methods.get_all_converters()]
ALL_INTERFACE_IMPLEMENTATIONS = [
    candidate[0].identifier
    for m in merge_methods.get_all()
    if m.interface is not None
    for candidate in m.interface.candidates
]
INTERNAL_MERGE_METHODS = [
    merge_checkpointing.identifier,
]
HIDDEN_MERGE_METHODS = ALL_CONVERTERS + ALL_INTERFACE_IMPLEMENTATIONS + INTERNAL_MERGE_METHODS


temporary_merged_recipes: List[Tuple[str, Iterable[Any]]] = []
cached_mergers_count: int = 0
temporary_mergers_count: int = 0
prompt_executor: Optional[execution.PromptExecutor] = None


def patch_prompt_executor():
    patch_key = "__mecha_execute_original"
    if not hasattr(execution.PromptExecutor, patch_key):
        setattr(execution.PromptExecutor, patch_key, execution.PromptExecutor.execute)
        execution.PromptExecutor.execute = functools.partialmethod(prompt_executor_execute, __original_function=execution.PromptExecutor.execute)

    patch_key = "__mecha_reset_original"
    if not hasattr(execution.PromptExecutor, patch_key):
        setattr(execution.PromptExecutor, patch_key, execution.PromptExecutor.reset)
        execution.PromptExecutor.reset = functools.partialmethod(prompt_executor_reset, __original_function=execution.PromptExecutor.reset)


def prompt_executor_execute(self, *args, __original_function, **kwargs):
    global cached_mergers_count, temporary_mergers_count, prompt_executor
    prompt_executor = self
    cached_mergers_count = 0
    temporary_mergers_count = 0
    try:
        res = __original_function(self, *args, **kwargs)

        for cache_id, mm_cache in cache_nodes.merge_method_caches.copy().items():
            if not mm_cache.marked:
                del cache_nodes.merge_method_caches[cache_id]
            else:
                mm_cache.unmark()

        return res
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
            for v in (v or ()):
                if v in temporary_merged_objects and k in prompt_executor.caches.outputs.cache:
                    prompt_executor.caches.outputs.cache.pop(k)

    del k, v
    temporary_merged_recipes.clear()
    model_management.cleanup_models()
    gc.collect()
    model_management.soft_empty_cache()


def prompt_executor_reset(self, *args, __original_function, **kwargs):
    global temporary_merged_recipes
    temporary_merged_recipes.clear()
    cache_nodes.merge_method_caches.clear()
    return __original_function(self, *args, **kwargs)


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
    CATEGORY = "mecha"

    def execute(self, recipe: ComfyMechaRecipe):
        return sd_mecha.serialize(recipe.node),


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
    CATEGORY = "mecha"

    def execute(self, recipe_txt: str):
        return ComfyMechaRecipe(sd_mecha.deserialize(recipe_txt.split("\n"))),


class MechaConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe": ("MECHA_RECIPE",),
            },
            "optional": {
                "target_config_from_recipe_override": ("MECHA_RECIPE",),
                "target_config": ([c.identifier for c in sd_mecha.extensions.model_configs.get_all()],),
            }
        }

    RETURN_TYPES = "MECHA_RECIPE",
    RETURN_NAMES = "recipe",
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(self, recipe: ComfyMechaRecipe, **kwargs):
        if "target_config_from_recipe_override" in kwargs:
            config_object = kwargs["target_config_from_recipe_override"].node
        elif "target_config" in kwargs:
            config_object = kwargs["target_config"]
        else:
            raise ValueError(
                "Neither 'target_config_from_recipe_override' nor 'target_config' are in kwargs. "
                f"This shouldn't ever happen but if it somehow does, here are the kwargs: {kwargs}"
            )
        return dataclasses.replace(
            recipe,
            node=sd_mecha.convert(
                recipe.node,
                config_object,
            )
        ),


class MechaMerger:
    @classmethod
    def INPUT_TYPES(cls):
        all_torch_devices = get_all_torch_devices()
        main_torch_device = model_management.get_torch_device().type
        return {
            "required": {
                "recipe": ("MECHA_RECIPE",),
            },
            "optional": {
                "fallback_model": (["none"] + [f for f in folder_paths.get_filename_list("checkpoints") if f.endswith(".safetensors")], {
                    "default": "none",
                }),
                "default_merge_device": (["none"] + all_torch_devices, {
                    "default": main_torch_device,
                }),
                "default_merge_dtype": (["none"] + list(DTYPE_MAPPING.keys()), {
                    "default": "fp32",
                }),
                "output_device": (["none"] + all_torch_devices, {
                    "default": "cpu",
                }),
                "output_dtype": (["none"] + list(DTYPE_MAPPING.keys()), {
                    "default": "fp16",
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
                "omit_vae": ("BOOLEAN", {
                    "default": True,
                }),
                "temporary_merge": ("BOOLEAN", {
                    "default": True,
                }),
                "strict_weight_space": ("BOOLEAN", {
                    "default": True,
                }),
                "check_finite_output": ("BOOLEAN", {
                    "default": True,
                }),
                "omit_non_finite_inputs": ("BOOLEAN", {
                    "default": True,
                }),
                "memoize_intermediates": ("BOOLEAN", {
                    "default": True,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "recipe_txt")
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    @classmethod
    def IS_CHANGED(cls, temporary_merge, **_kwargs):
        global cached_mergers_count, temporary_mergers_count
        cached_mergers_count += int(not temporary_merge)
        temporary_mergers_count += int(temporary_merge)
        return ""

    def execute(
        self,
        recipe: ComfyMechaRecipe,
        fallback_model: str,
        output_device: str,
        output_dtype: str,
        default_merge_device: str,
        default_merge_dtype: str,
        total_buffer_size: str,
        threads: int,
        omit_vae: bool,
        temporary_merge: bool,
        strict_weight_space: bool,
        **kwargs,
    ):
        check_finite_output: bool = kwargs.get("check_finite_output", True)
        omit_non_finite_inputs: bool = kwargs.get("omit_non_finite_inputs", True)
        memoize_intermediates: bool = kwargs.get("memoize_intermediates", True)

        global temporary_merged_recipes, prompt_executor
        total_buffer_size = memory_to_bytes(total_buffer_size)
        cache = recipe.cache
        recipe = sd_mecha.value_to_node(recipe.node)

        recipe_to_serialize = recipe.accept(CleanRecipeVisitor())
        with open_graph(recipe_to_serialize) as graph:
            recipe_to_serialize = graph.finalize_root(
                model_config_preference=("singleton-mecha",),
                merge_space="weight" if strict_weight_space else None,
                merge_space_preference=("weight",) if not strict_weight_space else None,
                check_mandatory_keys=False,
                check_extra_keys=False,
            )

        try:
            recipe_txt = sd_mecha.serialize(recipe_to_serialize, finalize=False)
        except TypeError:
            recipe_txt = str(id(recipe))

        try:
            already_merged_index = [t[0] for t in temporary_merged_recipes].index(recipe_txt)
            return temporary_merged_recipes[already_merged_index][1]
        except ValueError:
            if temporary_merged_recipes:
                free_temporary_merges(prompt_executor)

        if omit_vae and "vae" in recipe_to_serialize.model_config.components():
            recipe = sd_mecha.omit_component(recipe, "vae")

        model_management.unload_all_models()
        state_dict = sd_mecha.merge(
            recipe=recipe,
            fallback_model=sd_mecha.model(fallback_model) if fallback_model != "none" else None,
            merge_device=default_merge_device if default_merge_device != "none" else None,
            merge_dtype=DTYPE_MAPPING[default_merge_dtype] if default_merge_dtype != "none" else None,
            output_device=output_device if output_device != "none" else None,
            output_dtype=DTYPE_MAPPING[output_dtype] if output_dtype != "none" else None,
            threads=threads if threads >= 0 else None,
            total_buffer_size=total_buffer_size,
            strict_merge_space="weight" if strict_weight_space else None,
            strict_mandatory_keys=False,
            check_extra_keys=False,
            check_finite_output=check_finite_output,
            omit_non_finite_inputs=omit_non_finite_inputs,
            memoize_intermediates=memoize_intermediates,
            tqdm=ComfyTqdm,
            cache=cache,
        )
        res = load_state_dict_guess_config(state_dict, embedding_directory=folder_paths.get_folder_paths("embeddings"), output_vae=not omit_vae)
        if res is None:
            raise RuntimeError("Could not merge recipe: Comfy did not recognize this diffusion model.")

        res = res[: 3]
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

        self.update_comfy = RunLastAtMostEvery(self.update_comfy, 1)

    def update(self):
        self.progress.update()
        self.update_comfy()

    def update_comfy(self):
        self.comfy_progress.update_absolute(self.progress.n, self.progress.total)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.close()

    def __getattr__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            return getattr(self.progress, item)


MERGE_SPACES_OPTIONAL_INPUT_TYPE = ["default"] + [
    m.identifier for m in sd_mecha.extensions.merge_spaces.get_all()
]


class MechaModelRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    [
                        f
                        for f in folder_paths.get_filename_list("checkpoints")
                        if f.endswith(".safetensors")
                    ],
                ),
                "model_config": (["auto"] + [
                    config.identifier for config in sd_mecha.extensions.model_configs.get_all_base()
                    if "blocks" not in config.identifier
                ],),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        model_path: str,
        model_config: str,
    ):
        if model_config == "auto":
            model_config = None
        return ComfyMechaRecipe(sd_mecha.model(model_path, config=model_config)),


class MechaAnyModelRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_all_file_paths(),),
                "model_config": (["auto"] + [config.identifier for config in sd_mecha.extensions.model_configs.get_all()],),
            },
            "optional": {
                "merge_space": (MERGE_SPACES_OPTIONAL_INPUT_TYPE, {
                    "default": "default",
                })
            }
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        model_path: str,
        model_config: str,
        merge_space: str,
    ):
        if merge_space == "default":
            merge_space = "weight"
        if model_config == "auto":
            model_config = None
        return ComfyMechaRecipe(sd_mecha.model(model_path, config=model_config, merge_space=merge_space)),


class MechaAlreadyLoadedModelRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        model: comfy.model_patcher.ModelPatcher,
        clip: comfy.sd.CLIP = None,
        vae: comfy.sd.VAE = None
    ):
        clip_sd = None
        load_models = [model]
        if clip is not None:
            load_models.append(clip.load_model())
            clip_sd = clip.get_sd()
        vae_sd = None
        if vae is not None:
            vae_sd = vae.get_sd()

        model_management.load_models_gpu(load_models)
        lazy_casting_params = model.state_dict_for_saving(clip_sd, vae_sd)

        @sd_mecha.merge_method(register=False)
        def load_lazy_casting_param(
            _: sd_mecha.Parameter(torch.Tensor),
            **kwargs,
        ) -> sd_mecha.Return(torch.Tensor):
            key = kwargs["key"]
            return lazy_casting_params[key].to()

        sd = load_lazy_casting_param({
            key: tensor.as_subclass(torch.Tensor)
            for key, tensor in lazy_casting_params.items()
        })

        return ComfyMechaRecipe(sd),


class MechaLoraRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ([f for f in folder_paths.get_filename_list("loras") if f.endswith(".safetensors")],),
                "model_config": (["auto"] + [
                    config.identifier for config in sd_mecha.extensions.model_configs.get_all_aux()
                ],),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        model_path: str,
        model_config: str,
    ):
        if model_config == "auto":
            model_config = None

        recipe = sd_mecha.model(model_path, config=model_config)
        with open_graph(recipe, root_only=True, solve_merge_space=False) as graph:
            candidates = list(graph.root_candidates().model_config)
            if any(candidate.identifier in ("sdxl-kohya_kohya_lora", "sdxl-kohya_but_diffusers_kohya_lora") for candidate in candidates):
                recipe = sd_mecha.convert(recipe, "sdxl-sgm")
            elif any(candidate.identifier == "sd1-kohya_kohya_lora" for candidate in candidates):
                recipe = sd_mecha.convert(recipe, "sd1-ldm")
            else:
                raise RuntimeError(f"unsupported lora model config: {recipe.model_config.identifier}")

        return ComfyMechaRecipe(recipe),


def register_merge_methods():
    for method in merge_methods.get_all():
        if method.identifier in HIDDEN_MERGE_METHODS:
            continue

        method_name = method.get_identifier()
        class_name = f"{snake_case_to_upper(method_name)}MechaRecipe"
        short_title_name = snake_case_to_title(method_name)

        if method_name == "model_stock":
            method_name = "model_stock_for_tensor"

        title_name = f"{snake_case_to_title(method_name)} Mecha Recipe"
        NODE_CLASS_MAPPINGS[title_name] = make_comfy_node_class(class_name, method)
        NODE_DISPLAY_NAME_MAPPINGS[title_name] = short_title_name


def make_comfy_node_class(class_name: str, method: merge_methods.MergeMethod) -> type:
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
                    f"{name} ({default_args.args[default_index]})": ("MECHA_RECIPE", {
                        "default": default_args.args[default_index],
                    })
                    for default_index, name in enumerate(param_names.args[len_mandatory_args:])
                },
                **({
                    f"{param_names.vararg} ({'|'.join(sorted(merge_spaces.get_identifiers(input_merge_spaces.vararg)))})": ("MECHA_RECIPE_LIST", {
                        "default": [],
                    }),
                } if param_names.has_varargs() else {}),
                **{
                    f"{name} ({default_args.kwargs[index]})": ("MECHA_RECIPE", {
                        "default": default_args.kwargs[index],
                    })
                    for index, name in sorted(param_names.kwargs.items(), key=lambda t: t[0])
                    if index in default_args.kwargs
                },
                **({"cache": ("MECHA_MERGE_METHOD_CACHE",)} if method.create_cache() is not None else {}),
                "merge_checkpointing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Speeds up an entire branch of a merge graph that does not change often "
                               "in exchange of memory.\n\n"
                               "- true: store the first output of this recipe node on cpu memory in fp16. "
                               "On subsequent workflow executions, as long as the inputs do not change, "
                               "the cached keys are returned after being cast to the original device and dtype.\n"
                               "- false: do not store the output. "
                               "The recipe and its inputs will re-execute on subsequent workflow executions.\n\n"
                               "Note that the memory used to checkpoint the output is distinct from the cache feature. "
                               "In general, "
                               "you probably want to either use this *or* a cache unit, but not both at the same time "
                               "because the memory adds up.\n\n"
                               "The difference between merge checkpointing and cache is that merge checkpointing "
                               "completely re-merges from scratch if any input changes. "
                               "Merge checkpointing is also generally much faster than cache in the fast path.",
                }),
            },
        },
        "RETURN_TYPES": ("MECHA_RECIPE",),
        "RETURN_NAMES": ("recipe",),
        "FUNCTION": "execute",
        "OUTPUT_NODE": False,
        "CATEGORY": "mecha",
        "execute": get_method_node_execute(method),
    })


def fix_param_name(name):
    # for workflow backwards compatibility
    if name.endswith("_dict"):
        return name[:-len("_dict")]
    return name


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
    CATEGORY = "mecha"

    def execute(
        self,
        count: int,
        **kwargs,
    ):
        return [
            kwargs[f"recipe_{i}"]
            for i in range(count) if f"recipe_{i}" in kwargs
        ],


class MechaSubtractRecipeList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 2, "min": 0, "max": MAX_VARARGS_MODELS, "step": 1}),
                "base_recipe": ("MECHA_RECIPE",),
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
    CATEGORY = "mecha"

    def execute(
        self,
        count: int,
        base_recipe: ComfyMechaRecipe,
        **kwargs,
    ):
        return [
            ComfyMechaRecipe(
                kwargs[f"recipe_{i}"].node - base_recipe.node,
                kwargs[f"recipe_{i}"].cache | base_recipe.cache,
            )
            for i in range(count) if f"recipe_{i}" in kwargs
        ],


def get_all_torch_devices() -> List[str]:
    torch_device = model_management.get_torch_device().type
    return [
        "cpu",
        *([torch_device] if torch_device != "cpu" else []),
        *([f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []),
    ]


def get_method_node_execute(method: merge_methods.MergeMethod):
    param_names = method.get_param_names()
    param_defaults = method.get_default_args()
    num_mandatory_args = len(param_names.args) - len(param_defaults.args)

    def execute(*_args, **kwargs):
        global merge_checkpointing
        # private key for caching
        cache_node = kwargs.pop("cache", None)
        use_merge_checkpointing = kwargs.pop("merge_checkpointing")

        # remove default values / merge space from keys
        # comfy nodes cannot distinguish display names from id names
        # as a result we have to unmangle things here
        for k in list(kwargs):
            if " (" in k and k.endswith(")"):
                new_k = k.split(" ")[0]
                kwargs[new_k] = kwargs.pop(k)

        cache_all = functools.reduce(operator.or_, (collect_cache(v) for v in kwargs.values()), {})
        args = [
            kwargs.pop(k).node
            if i < num_mandatory_args else
            kwargs.pop(k, ComfyMechaRecipe(param_defaults.args[i - num_mandatory_args])).node
            for i, k in enumerate(param_names.args)
        ]
        if param_names.has_varargs():
            args += (v.node for v in kwargs.pop(param_names.vararg, ()))
        kwargs = {
            k: v.node
            for k, v in kwargs.items()
        }

        node = method(*args, **kwargs)
        if method.identifier == "add_difference" and args:
            node = node | args[0]

        if cache_node is not None:
            if cache_node.setdefault("__merge_method_identifier", method.identifier) != method.identifier:
                raise ValueError("Cache Units cannot be reused with different types of merge methods. Recreate the Cache Unit node or plug in one that is already compatible.")
            cache_all[node] = cache_node

        if use_merge_checkpointing:
            node = merge_checkpointing(node)
            cache_all[node] = {}

        return ComfyMechaRecipe(node, cache_all),

    def collect_cache(recipe):
        if isinstance(recipe, ComfyMechaRecipe):
            return recipe.cache
        if isinstance(recipe, list):
            return functools.reduce(operator.or_, (collect_cache(v) for v in value), {})
        return {}

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


def get_all_folder_paths():
    return [
        pathlib.Path(p)
        for item in get_folder_path_ids()
        for p in folder_paths.get_folder_paths(item)
    ]


def get_all_file_paths():
    return [
        p
        for item in get_folder_path_ids()
        for p in folder_paths.get_filename_list(item)
        if p.endswith(".safetensors")
    ]


def get_folder_path_ids() -> Tuple[str, ...]:
    return "checkpoints", "loras", "clip", "unet", "vae", "embeddings"


for p in get_all_folder_paths():
    if p not in sd_mecha.extensions.model_dirs.get_all():
        sd_mecha.extensions.model_dirs.add_path(p)


class CleanRecipeVisitor(sd_mecha.recipe_nodes.RecipeVisitor):
    def visit_literal(self, node: LiteralRecipeNode):
        new_dict = {}
        changed = False
        for k, v in node.value_dict.items():
            if isinstance(v, RecipeNode):
                new_dict[k] = v.accept(self)
                changed = True
            else:
                new_dict[k] = v

        if changed:
            return LiteralRecipeNode(
                new_dict,
                model_config=node.model_config,
                merge_space=node.merge_space,
            )
        else:
            return node

    def visit_model(self, node: ModelRecipeNode):
        return node

    def visit_merge(self, node: MergeRecipeNode):
        if node.merge_method.identifier == "merge_checkpointing":
            underlying_node = node.bound_args.arguments["a"]
            return underlying_node.accept(self)
        return MergeRecipeNode(
            node.merge_method,
            node.bound_args.signature.bind(
                *(v.accept(self) for v in node.bound_args.args),
                **{k: v.accept(self) for k, v in node.bound_args.kwargs.items()},
            ),
            node.model_config,
            node.merge_space,
        )


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
    "Mecha Converter": MechaConverter,
    "Model Mecha Recipe": MechaModelRecipe,
    "Any Model Mecha Recipe": MechaAnyModelRecipe,
    "Already Loaded Model Mecha Recipe": MechaAlreadyLoadedModelRecipe,
    "Lora Mecha Recipe": MechaLoraRecipe,
    "Mecha Recipe List": MechaRecipeList,
    "Mecha Subtract Recipe List": MechaSubtractRecipeList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mecha Merger": "Merger",
    "Mecha Serializer": "Serializer",
    "Mecha Deserializer": "Deserializer",
    "Mecha Converter": "Converter",
    "Model Mecha Recipe": "Model",
    "Any Model Mecha Recipe": "Any Model",
    "Already Loaded Model Mecha Recipe": "Already Loaded Model",
    "Lora Mecha Recipe": "Lora",
    "Mecha Recipe List": "Recipe List",
    "Mecha Subtract Recipe List": "Subtract Recipe List",
}

register_merge_methods()
