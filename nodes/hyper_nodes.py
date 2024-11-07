import sd_mecha


class BlocksMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (["custom"], {
                    "default": "custom",
                }),
                "blocks": ("STRING", {
                    "default": "",
                }),
                "validate_num_blocks": ("BOOLEAN", {
                    "default": True,
                }),
                "default": ("FLOAT", {
                    "default": 0.0,
                    "min": -2**64,
                    "max": 2**64,
                    "step": 0.01,
                }),
                "model_config": ([x.identifier for x in sd_mecha.extensions.model_config.get_all()],),
                "model_component": (["unet", "txt", "txt2", "t5xxl"], {
                    "default": "unet",
                }),
            },
        }

    RETURN_TYPES = ("MECHA_HYPER",)
    RETURN_NAMES = ("hyper",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        preset: str,
        blocks: str,
        validate_num_blocks: bool,
        default: float,
        model_config: str,
        model_component: str,
    ):
        if preset != "custom":
            blocks = ""
            validate_num_blocks = True
            default = 0.0

        try:
            return sd_mecha.default(
                model_config,
                value=default,
            ) | sd_mecha.blocks(
                model_config,
                model_component if model_component else None,
                *((
                      float(block.strip()) if block.strip() else default
                      for block in blocks.split(",")
                  ) if blocks.strip() else ()),
                strict=validate_num_blocks,
            ),
        except ValueError as e:
            raise ValueError(f"Wrong number of blocks for model architecture '{model_config}'") from e


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

    RETURN_TYPES = ("MECHA_HYPER",)
    RETURN_NAMES = ("hyper",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        value: float,
    ):
        return value,


def register_defaults_hyper_nodes():
    for config in sd_mecha.extensions.model_config.get_all():
        class_name = f"{config.identifier.upper()}DefaultsHyper"
        title_name = f"{config.identifier} Defaults Hyper"
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
        "RETURN_TYPES": ("MECHA_HYPER",),
        "RETURN_NAMES": ("hyper",),
        "FUNCTION": "execute",
        "OUTPUT_NODE": False,
        "CATEGORY": "advanced/model_merging/mecha",
        "execute": get_defaults_hyper_node_execute(config),
    })


def get_defaults_hyper_node_execute(config: sd_mecha.extensions.model_config.ModelConfig):
    def execute(self, **kwargs):
        res = {}
        for component in config.components:
            res = res | sd_mecha.default(config.identifier, component, kwargs[component])
        return res,
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
