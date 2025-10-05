import functools
import operator
import re
import typing
import sd_mecha
from typing import List, Literal
from sd_mecha.extensions import model_configs


BLOCK_CONFIGS = {
    "sd1-ldm": model_configs.resolve("sd1-supermerger_blocks"),
    "sdxl-sgm": model_configs.resolve("sdxl-supermerger_blocks"),
}
MAX_BLOCKS = max(len(config.keys()) for config in BLOCK_CONFIGS.values())


class BlocksMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (["custom"], {
                    "default": "custom",
                }),
                "blocks": ("STRING", {
                    "default": ","*(MAX_BLOCKS-1),
                }),
                "default": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                    "step": 0.01,
                }),
                "model_config": (list(BLOCK_CONFIGS),),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        preset: str,
        blocks: str,
        default: float,
        model_config: str,
    ):
        if preset != "custom":
            blocks = ""
            default = 0.0

        block_config = BLOCK_CONFIGS[model_config]
        return sd_mecha.convert(
            {
                block_name: float(block.strip()) if block.strip() else default
                for block_name, block in zip(block_config.keys(), blocks.split(","))
            } if blocks.strip() else {},
            model_config,
        ) | default,


RegexMode = Literal["simple", "regex"]
REGEX_MODES = typing.get_args(RegexMode)


class RegexWeightsMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regex_weights": ("STRING", {
                    "default": "",
                    "tooltip": 'one regex per line:\n\n<pattern>: <weight>\n<pattern>: <weight>',
                    "multiline": True,
                }),
                "default": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                    "step": 0.01,
                }),
                "model_config": (list(config.identifier for config in sd_mecha.extensions.model_configs.get_all()), {
                    "default": "sdxl-sgm"
                }),
                "regex_mode": (REGEX_MODES, {
                    "default": REGEX_MODES[0],
                    "tooltip": '- simple: the only special character is *, which means to match any number of characters. For example, "a*n" will match "alliteration". \\* matches * literally and \\\\ matches \\ literally\n'
                               '- regex: full python regex mode. for example, ".*" matches any number of characters'
                }),
                "match_full_keys": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "- True: a pattern needs to cover a key from start to end to match it\n"
                               "- False: a pattern only needs cover a part of a key to match it"
                }),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE", "STRING")
    RETURN_NAMES = ("recipe", "recipe_txt")
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        regex_weights: str,
        default: float,
        model_config: str,
        regex_mode: RegexMode,
        match_full_keys: bool,
    ):
        model_config = sd_mecha.extensions.model_configs.resolve(model_config)
        weights_str = [weight_regex for weight_regex in regex_weights.split("\n") if weight_regex.strip()]
        key_matcher = KeyMatcher(weights_str, mode=regex_mode, full_match=match_full_keys)
        weights = {}
        for key in model_config.keys():
            weight = key_matcher.get_weight_for(key)
            if weight is None:
                continue
            weights[key] = weight

        if weights:
            recipe = sd_mecha.literal(weights, model_config) | default
        else:
            recipe = sd_mecha.literal(default, model_config)

        return recipe, sd_mecha.serialize(recipe)


class KeyMatcher:
    def __init__(self, matchers: List[str], mode: RegexMode, full_match: bool):
        self.matchers = []
        self.full_match = full_match
        for matcher in matchers:
            parts = matcher.rsplit(":", maxsplit=1)
            if len(parts) != 2:
                continue
            pattern, weight = parts
            try:
                weight = float(weight.strip())
                pattern = self._compile(pattern.strip(), mode)
                self.matchers.append((pattern, weight))
            except ValueError:
                continue

    @staticmethod
    def _compile(pattern: str, mode: str):
        if mode == "simple":
            processed_pattern = ""
            current_part = []
            is_escape = False

            def flush_current_part():
                nonlocal current_part
                part = re.escape("".join(current_part))
                current_part.clear()
                return part

            for char in pattern:
                if is_escape:
                    current_part.append(char)
                    is_escape = False
                elif char == "\\":
                    is_escape = True
                elif char == "*":
                    processed_pattern += flush_current_part() + ".*"
                    current_part.clear()
                else:
                    current_part.append(char)
            processed_pattern += flush_current_part()
            return re.compile(processed_pattern)
        elif mode == "regex":
            return re.compile(pattern)
        else:
            raise ValueError(f"unrecognized mode '{mode}'")

    def get_weight_for(self, key):
        if self.full_match:
            match_fn = re.fullmatch
        else:
            match_fn = re.search

        for matcher, weight in self.matchers:
            if match_fn(matcher, key):
                return weight

        return None


class SdxlBlocksMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "BASE": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                    "step": 0.01,
                }),
                **{
                    f"IN{i:02}": ("FLOAT", {
                        "default": 0.0,
                        "min": -2**64,
                        "max": 2**64,
                        "step": 0.01,
                    })
                    for i in range(9)
                },
                "M00": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                    "step": 0.01,
                }),
                **{
                    f"OUT{i:02}": ("FLOAT", {
                        "default": 0.0,
                        "min": -2**64,
                        "max": 2**64,
                        "step": 0.01,
                    })
                    for i in range(9)
                },
                "VAE": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                    "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        **kwargs,
    ):
        blocks = sd_mecha.literal(kwargs, "sdxl-supermerger_blocks")
        blocks = sd_mecha.convert(blocks, "sdxl-sgm")
        return blocks,


class Sd1BlocksMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "BASE": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                    "step": 0.01,
                }),
                **{
                    f"IN{i:02}": ("FLOAT", {
                        "default": 0.0,
                        "min": -2**64,
                        "max": 2**64,
                        "step": 0.01,
                    })
                    for i in range(12)
                },
                "M00": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                    "step": 0.01,
                }),
                **{
                    f"OUT{i:02}": ("FLOAT", {
                        "default": 0.0,
                        "min": -2**64,
                        "max": 2**64,
                        "step": 0.01,
                    })
                    for i in range(12)
                },
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        **kwargs,
    ):
        blocks = sd_mecha.literal(kwargs, "sd1-supermerger_blocks")
        blocks = sd_mecha.convert(blocks, "sd1-ldm")
        return blocks,


class FloatMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                    "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        value: float,
    ):
        return value,


class IntMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 0,
                    "min": -2**64,
                    "max": 2**64,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        value: int,
    ):
        return value,


class StringMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING",),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        value: str,
    ):
        return value,


class BoolMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("BOOLEAN",),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"

    def execute(
        self,
        value: bool,
    ):
        return value,


def register_components_params_nodes():
    for config in model_configs.get_all():
        identifier = config.identifier
        if identifier in ["sdxl-sgm", "sd1-ldm", "flux-flux"]:  # for backwards compatibility
            identifier = identifier.split("-")[0]
        class_name = f"{identifier.upper()}DefaultsHyper"
        title_name = f"{identifier.upper()} Defaults Hyper"
        NODE_CLASS_MAPPINGS[title_name] = make_components_params_node_class(class_name, config)

        display_name = f"{config.identifier.upper()} Components Params"
        NODE_DISPLAY_NAME_MAPPINGS[title_name] = display_name


def make_components_params_node_class(class_name: str, config: model_configs.ModelConfig) -> type:
    return type(class_name, (object,), {
        "INPUT_TYPES": lambda: {
            "required": {
                **{
                    component: ("FLOAT", {
                        "default": 0.0,
                        "min": -2**64,
                        "max": 2**64,
                        "step": 0.01,
                    })
                    for component in list(config.components())
                },
            },
        },
        "RETURN_TYPES": ("MECHA_RECIPE",),
        "RETURN_NAMES": ("recipe",),
        "FUNCTION": "execute",
        "OUTPUT_NODE": False,
        "CATEGORY": "mecha",
        "execute": get_components_params_node_execute(config),
    })


def get_components_params_node_execute(config: model_configs.ModelConfig):
    def execute(self, **kwargs):
        recipes = [
            sd_mecha.pick_component(sd_mecha.literal(kwargs[component_id], config), component_id)
            for component_id, component in config.components().items()
        ]
        return functools.reduce(operator.or_, recipes),
    return execute


NODE_CLASS_MAPPINGS = {
    "Blocks Mecha Hyper": BlocksMechaHyper,
    "SDXL-SGM Mecha Blocks Parameters": SdxlBlocksMechaHyper,
    "SD1-LDM Mecha Blocks Parameters": Sd1BlocksMechaHyper,
    "Mecha Regex Weights": RegexWeightsMechaHyper,
    "Int Mecha Hyper": IntMechaHyper,
    "Float Mecha Hyper": FloatMechaHyper,
    "String Mecha Hyper": StringMechaHyper,
    "Bool Mecha Hyper": BoolMechaHyper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Blocks Mecha Hyper": "Blocks",
    "SDXL-SGM Mecha Blocks Parameters": "SDXL-SGM Blocks",
    "SD1-LDM Mecha Blocks Parameters": "SD1-LDM Blocks",
    "Mecha Regex Weights": "Regex Weights",
    "Float Mecha Hyper": "Float",
    "Int Mecha Hyper": "Int",
    "String Mecha Hyper": "String",
    "Bool Mecha Hyper": "Bool",
}

register_components_params_nodes()
