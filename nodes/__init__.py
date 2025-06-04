from . import merge_nodes
from . import parameter_nodes
from . import cache_nodes

NODE_CLASS_MAPPINGS = (
    merge_nodes.NODE_CLASS_MAPPINGS |
    parameter_nodes.NODE_CLASS_MAPPINGS |
    cache_nodes.NODE_CLASS_MAPPINGS
)
NODE_DISPLAY_NAME_MAPPINGS = (
    merge_nodes.NODE_DISPLAY_NAME_MAPPINGS |
    parameter_nodes.NODE_DISPLAY_NAME_MAPPINGS |
    cache_nodes.NODE_DISPLAY_NAME_MAPPINGS
)
