import functools
import operator

import sd_mecha
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
                }),
                "model_config": (list(BLOCK_CONFIGS),),
            },
        }

    RETURN_TYPES = ("MECHA_RECIPE",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

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
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        value: float,
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
    CATEGORY = "advanced/model_merging/mecha"

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
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        value: bool,
    ):
        return value,


def register_components_params_nodes():
    for config in model_configs.get_all():
        class_name = f"{config.identifier.upper()}ComponentsParams"
        title_name = f"{config.identifier} Components Params"
        NODE_CLASS_MAPPINGS[title_name] = make_components_params_node_class(class_name, config)


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
        "CATEGORY": "advanced/model_merging/mecha",
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
    "Float Mecha Hyper": FloatMechaHyper,
    "String Mecha Hyper": StringMechaHyper,
    "Bool Mecha Hyper": BoolMechaHyper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Blocks Mecha Hyper": "Blocks",
    "Float Mecha Hyper": "Float",
    "String Mecha Hyper": "String",
    "Bool Mecha Hyper": "Bool",
}

register_components_params_nodes()
