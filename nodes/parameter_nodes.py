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


class SdxlBlocksMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "BASE": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                }),
                **{
                    f"IN{i:02}": ("FLOAT", {
                        "default": 0.0,
                        "min": -2**64,
                        "max": 2**64,
                    })
                    for i in range(9)
                },
                "M00": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                }),
                **{
                    f"OUT{i:02}": ("FLOAT", {
                        "default": 0.0,
                        "min": -2**64,
                        "max": 2**64,
                    })
                    for i in range(9)
                },
                "VAE": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
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
                }),
                **{
                    f"IN{i:02}": ("FLOAT", {
                        "default": 0.0,
                        "min": -2**64,
                        "max": 2**64,
                    })
                    for i in range(12)
                },
                "M00": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                }),
                **{
                    f"OUT{i:02}": ("FLOAT", {
                        "default": 0.0,
                        "min": -2**64,
                        "max": 2**64,
                    })
                    for i in range(12)
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
    CATEGORY = "advanced/model_merging/mecha"

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
    CATEGORY = "advanced/model_merging/mecha"

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
    "SDXL-SGM Mecha Blocks Parameters": SdxlBlocksMechaHyper,
    "SD1-LDM Mecha Blocks Parameters": Sd1BlocksMechaHyper,
    "Int Mecha Hyper": IntMechaHyper,
    "Float Mecha Hyper": FloatMechaHyper,
    "String Mecha Hyper": StringMechaHyper,
    "Bool Mecha Hyper": BoolMechaHyper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Blocks Mecha Hyper": "Blocks",
    "SDXL-SGM Mecha Blocks Parameters": "SDXL-SGM Blocks",
    "SD1-LDM Mecha Blocks Parameters": "SD1-LDM Blocks",
    "Float Mecha Hyper": "Float",
    "Int Mecha Hyper": "Int",
    "String Mecha Hyper": "String",
    "Bool Mecha Hyper": "Bool",
}

register_components_params_nodes()
