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
                    "step": 0.01,
                }),
                "model_arch": (sd_mecha.extensions.model_arch.get_all(),),
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
        model_arch: str,
        model_component: str,
    ):
        if preset != "custom":
            blocks = ""
            validate_num_blocks = True
            default = 0.0

        try:
            return sd_mecha.default(
                model_arch=model_arch,
                value=default,
            ) | sd_mecha.blocks(
                model_arch,
                model_component if model_component else None,
                *((
                    float(block.strip()) if block.strip() else default
                    for block in blocks.split(",")
                ) if blocks.strip() else ()),
                strict=validate_num_blocks,
            ),
        except ValueError as e:
            raise ValueError(f"Wrong number of blocks for model architecture '{model_arch}'") from e


class FloatMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
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
    for arch_id in sd_mecha.extensions.model_arch.get_all():
        arch = sd_mecha.extensions.model_arch.resolve(arch_id)
        class_name = f"{arch_id.upper()}DefaultsHyper"
        title_name = f"{arch_id.upper()} Defaults Hyper"
        NODE_CLASS_MAPPINGS[title_name] = make_defaults_hyper_node_class(class_name, arch)


def make_defaults_hyper_node_class(class_name: str, arch: sd_mecha.extensions.model_arch.ModelArch) -> type:
    return type(class_name, (object,), {
        "INPUT_TYPES": lambda: {
            "required": {
                **{
                    component: ("FLOAT", {
                        "default": 0.0,
                    })
                    for component in arch.components
                },
            },
        },
        "RETURN_TYPES": ("MECHA_HYPER",),
        "RETURN_NAMES": ("hyper",),
        "FUNCTION": "execute",
        "OUTPUT_NODE": False,
        "CATEGORY": "advanced/model_merging/mecha",
        "execute": get_defaults_hyper_node_execute(arch),
    })


def get_defaults_hyper_node_execute(arch: sd_mecha.extensions.model_arch.ModelArch):
    def execute(self, **kwargs):
        res = {}
        for component in arch.components:
            res = res | sd_mecha.default(arch.identifier, component, kwargs[component])
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
