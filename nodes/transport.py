import dataclasses
from typing import Optional
from sd_mecha import RecipeNodeOrValue


@dataclasses.dataclass
class ComfyMechaRecipe:
    node: RecipeNodeOrValue
    cache: Optional[dict] = dataclasses.field(default_factory=dict)
