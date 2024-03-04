import sd_mecha


class DefaultMechaHyper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0}),
                "model_arch": (sd_mecha.extensions.model_arch.get_all(),),
                "model_component": ("STRING", {"default": ""}),
            },
        }
    RETURN_TYPES = ("HYPER",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "advanced/model_merging/mecha"

    def execute(
        self,
        value: float,
        model_arch: str,
        model_component: str,
    ):
        return sd_mecha.default(
            model_arch=model_arch,
            model_components=[model_component] if model_component else None,
            value=value,
        )


NODE_CLASS_MAPPINGS = {
    "Default Mecha Hyper": DefaultMechaHyper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Default Mecha Hyper": "Mecha Hyper",
}
