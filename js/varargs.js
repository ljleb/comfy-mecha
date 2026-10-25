// original src: https://github.com/jags111/efficiency-nodes-comfyui
import { app } from "../../scripts/app.js";

let customDefInputNames = {};

const MAX_VARARGS_MODELS = 64;  // arbitrary limit to n-models methods (open an issue if this is a problem)

function handleMechaModelListVisibilityByCount(node, visibleCount, _widget, countOffset=0) {
    if (node.origInputs === undefined) {
        node.origInputs = node.inputs;
    }

    for (let i = 0; i < node.inputs.length; ++i) {
        const input = node.inputs[i];
        const origInput = node.origInputs[i];

        for (const key of Object.keys(input)) {
            origInput[key] = input[key];
        }
    }

    node.inputs = Array.from(node.origInputs);
    node.inputs.length = visibleCount + countOffset;
    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

function handleMechaHyperBlocksVisibilityByPreset(node, preset) {
    // Hide the preset-specific widgets in place (type = "hidden") rather than splicing them
    // out of node.widgets. Nothing is removed from the array, so no master copy is needed and
    // switching back to "custom" simply restores each widget's original type.
    const hiddenNames = ["blocks", "validate_num_blocks", "default"];
    for (const w of node.widgets || []) {
        if (!hiddenNames.includes(w.name)) continue;
        if (preset !== "custom") {
            if (w._mechaOrigType === undefined) w._mechaOrigType = w.type;
            w.type = "hidden";
        } else if (w._mechaOrigType !== undefined) {
            w.type = w._mechaOrigType;
            w._mechaOrigType = undefined;
        }
    }

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

function handleMechaConverterVisibility(node, input_link) {
    if (input_link !== null) {
        // Hide widgets in place. Capture the original types once so we can restore them.
        // Mutating widget.type avoids swapping node.widgets, which breaks on frontends where
        // node.widgets is a reactive proxy (the saved reference ends up empty and won't render).
        if (!node._mechaHidden) {
            node._mechaHidden = true;
            node._mechaWidgetTypes = (node.widgets || []).map((w) => w.type);
            for (const w of node.widgets || []) {
                w.type = "hidden";
            }
        }
    } else if (node._mechaHidden) {
        const types = node._mechaWidgetTypes;
        node._mechaHidden = false;
        node._mechaWidgetTypes = undefined;
        for (let i = 0; i < (node.widgets || []).length && i < types.length; ++i) {
            if (node.widgets[i]) node.widgets[i].type = types[i];
        }
    }

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

function handleMechaMergeMethodCacheInitByIdentifier(node, identifier, widget) {
    setTimeout(() => {
        widget.type = "hidden";
        widget.value = randomCacheId();
    }, 0);
}

function handleMechaConverterVisibilityByConnection(node, input) {
    handleMechaConverterVisibility(node, input.link);
}

function widgetLogic(node, widget, value) {
    const handler = nodeWidgetHandlers[node.comfyClass || node.type]?.[widget.name];
    if (handler) {
        handler(node, value, widget);
    }
}

function inputLogic(node, input) {
    const handler = nodeInputHandlers[node.comfyClass || node.type]?.[input.name];
    if (handler) {
        handler(node, input);
    }
}

const nodeWidgetHandlers = {
    "Mecha Recipe List": {
        "count": handleMechaModelListVisibilityByCount
    },
    "Mecha Subtract Recipe List": {
        "count": (...args) => handleMechaModelListVisibilityByCount(...args, 1)
    },
    "Blocks Mecha Hyper": {
        "preset": handleMechaHyperBlocksVisibilityByPreset
    },
    "Mecha Merge Method Cache Unit": {
        "identifier": handleMechaMergeMethodCacheInitByIdentifier
    },
};

const nodeInputHandlers = {
    "Mecha Converter": {
        "target_config_from_recipe_override": handleMechaConverterVisibilityByConnection
    },
};

function randomCacheId() {
    const letters = "abcdefghijklmnopqrstuvwxyz";
    let str = "";
    for (let i = 0; i < 64; i++) {
        const idx = Math.floor(Math.random() * letters.length);
        str += letters[idx];
    }
    return str;
}

app.registerExtension({
    name: "mecha.widgethider",
    nodeCreated(node) {
        for (const w of node.widgets || []) {
            widgetLogic(node, w, w.value);

            const original_callback = w.callback?.bind?.(w);
            w.callback = (value) => {
                original_callback?.(value);
                widgetLogic(node, w, value);
            }
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.output[0] === "MECHA_RECIPE" && nodeData.input) {
            const inputs = Object.assign({}, nodeData.input.required || {}, nodeData.input.optional || {});
            for (const input_name in inputs) {
                if (["MECHA_RECIPE", "MECHA_RECIPE_LIST", "MECHA_MERGE_METHOD_CACHE"].includes(inputs[input_name][0])) {
                    if (!(nodeData.name in customDefInputNames)) {
                        customDefInputNames[nodeData.name] = {};
                    }
                    customDefInputNames[nodeData.name][input_name] = (inputs[input_name][1] || {}).name || input_name;
                }
            }
        }

        if (nodeData.name in nodeInputHandlers) {
            const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (side, slot, connect, link_info, output) {
                if (side === 1 && this.inputs?.[slot]) {
                    inputLogic(this, this.inputs[slot]);
                }
                return originalOnConnectionsChange?.apply(this, arguments);
            };
        }
    },
    loadedGraphNode(node, app) {
        for (const iv of (node.inputs || []).slice()) {
            const iv_idx = node.inputs.indexOf(iv);
            if (iv.type === "MECHA_RECIPE" || iv.type === "MECHA_RECIPE_LIST" || iv.type === "MECHA_HYPER") {
                for (const iv2 of node.inputs.slice()) {
                    if (
                        iv.name.split(" ")[0] === iv2.name.split(" ")[0] &&
                        iv !== iv2 &&
                        (customDefInputNames[node.type] || {})[iv2.name] === iv2.name
                    ) {
                        if (iv.link !== null) {
                            iv2.link = iv.link;
                        }
                        node.inputs.splice(iv_idx, 1);
                        const newHeight = node.computeSize()[1];
                        node.setSize([node.size[0], newHeight]);
                    }
                }
            }
        }

        for (const iv of (node.inputs || []).slice()) {
            inputLogic(node, iv);
        }
    },
});
