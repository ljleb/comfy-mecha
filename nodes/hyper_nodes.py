import sd_mecha


BLOCK_CONFIGS = {
    "sd1-ldm": sd_mecha.extensions.model_config.resolve("sd1_blocks-supermerger"),
    "sdxl-sgm": sd_mecha.extensions.model_config.resolve("sdxl_blocks-supermerger"),
}
MAX_BLOCKS = max(len(config.keys) for config in BLOCK_CONFIGS.values())


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
                for block_name, block in zip(block_config.keys.keys(), blocks.split(","))
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


def register_defaults_hyper_nodes():
    for config in sd_mecha.extensions.model_config.get_all():
        class_name = f"{config.identifier.upper()}DefaultsHyper"
        title_name = f"{config.identifier} Default Params"
        NODE_CLASS_MAPPINGS[title_name] = make_defaults_hyper_node_class(class_name, config)


def make_defaults_hyper_node_class(class_name: str, config: sd_mecha.extensions.model_config.ModelConfig) -> type:
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
                    for component in sorted(list(config.components))
                },
            },
        },
        "RETURN_TYPES": ("MECHA_RECIPE",),
        "RETURN_NAMES": ("recipe",),
        "FUNCTION": "execute",
        "OUTPUT_NODE": False,
        "CATEGORY": "advanced/model_merging/mecha",
        "execute": get_defaults_hyper_node_execute(config),
    })


def get_defaults_hyper_node_execute(config: sd_mecha.extensions.model_config.ModelConfig):
    def execute(self, **kwargs):
        return sd_mecha.literal(
            {
                k: kwargs[component_id]
                for component_id, component in config.components.items()
                for k in component.keys
            },
            config.identifier,
        ),
    return execute


NODE_CLASS_MAPPINGS = {
    "Blocks Mecha Hyper": BlocksMechaHyper,
    "Float Mecha Hyper": FloatMechaHyper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Blocks Mecha Hyper": "Blocks",
    "Float Mecha Hyper": "Float",
}

register_defaults_hyper_nodes()
