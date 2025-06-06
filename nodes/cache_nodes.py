from typing import Dict
from sd_mecha.recipe_nodes import RecipeVisitor, MergeRecipeNode, ModelRecipeNode, LiteralRecipeNode


CACHE_OBJECT_KEY = "__merge_method_cache_object"


class MarkCachesVisitor(RecipeVisitor):
    def visit_literal(self, node: LiteralRecipeNode):
        pass

    def visit_model(self, node: ModelRecipeNode):
        pass

    def visit_merge(self, node: MergeRecipeNode):
        cache = node.cache.get(CACHE_OBJECT_KEY)
        if cache is not None:
            cache.mark()


class MergeMethodCache:
    def __init__(self, cache: dict = None):
        if cache is None:
            cache = {}
        cache[CACHE_OBJECT_KEY] = self
        self.cache = cache
        self.marked = False

    def mark(self):
        self.marked = True

    def unmark(self):
        self.marked = False


merge_method_caches: Dict[str, MergeMethodCache] = {}


class MechaMergeMethodCacheUnit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "identifier": ("STRING", {
                    "default": "",
                }),
            },
        }

    RETURN_TYPES = ("MECHA_MERGE_METHOD_CACHE",)
    RETURN_NAMES = ("cache",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "mecha"
    DESCRIPTION = ("Holds a persistent merge method cache. This node outputs a cache dict via its cache port; "
                   "merge methods can store or retrieve intermediate results (tensors, SVDs, etc.) in that dict."
                   "\n\n"
                   "To enable caching, connect this node's cache output to a recipe node's cache input. "
                   "Once a cache unit has been used by a given merge method type, it becomes tied to "
                   "that type of merge method "
                   "(e.g., you cannot share the same cache between Truncate Rank and Rotate). "
                   "On subsequent runs, merge methods will reuse cached artifacts. "
                   "Each merge method is responsible for deciding when some cache results need to be recomputed."
                   "\n\n"
                   "The cache persists as long as this node remains in the workflow, even if it is disconnected; "
                   "if you run a workflow that doesn't include this node, its cache is cleared after execution.\n\n"
                   "Recreating this node from the context menu button 'Fix node (recreate)' resets the cache. "
                   "Copy pasting the node creates a cache unit that refers to a different memory unit.")

    @classmethod
    def IS_CHANGED(cls, identifier):
        if identifier is None:
            return ""

        cache = merge_method_caches.setdefault(identifier, MergeMethodCache())
        cache.mark()

        return ""

    @classmethod
    def execute(cls, identifier):
        return merge_method_caches.setdefault(identifier, MergeMethodCache()).cache,


NODE_CLASS_MAPPINGS = {
    "Mecha Merge Method Cache Unit": MechaMergeMethodCacheUnit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mecha Merge Method Cache Unit": "Cache Unit",
}
