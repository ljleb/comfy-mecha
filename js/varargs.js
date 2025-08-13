// original src: https://github.com/jags111/efficiency-nodes-comfyui
import { app } from "../../scripts/app.js";

let customDefInputNames = {};

const MAX_VARARGS_MODELS = 64;  // arbitrary limit to n-models methods (open an issue if this is a problem)

const findWidgetIndexByName = (widgets, name) => {
    return widgets ? widgets.findIndex((w) => w.name === name) : null;
};

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
    if (node.origWidgets === undefined) {
        node.origWidgets = node.widgets;
    }

    for (let i = 0; i < node.widgets.length; ++i) {
        const widget = node.widgets[i];
        node.origWidgets[findWidgetIndexByName(node.origWidgets, widget.name)] = widget;
    }

    if (preset === "custom") {
        node.widgets = Array.from(node.origWidgets);
    } else {
        node.widgets = Array.from(node.origWidgets);

        const blocksWidgetIndex = findWidgetIndexByName(node.widgets, "blocks");
        node.widgets.splice(blocksWidgetIndex, 1);

        const validateNumBlocksWidgetIndex = findWidgetIndexByName(node.widgets, "validate_num_blocks");
        node.widgets.splice(validateNumBlocksWidgetIndex, 1);

        const defaultWidgetIndex = findWidgetIndexByName(node.widgets, "default");
        node.widgets.splice(defaultWidgetIndex, 1);
    }

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

function handleMechaConverterVisibility(node, input_link) {
    if (node.origWidgets === undefined) {
        node.origWidgets = node.widgets;
    }

    if (input_link !== null) {
        node.widgets = [];
    } else {
        node.widgets = node.origWidgets;
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
    const handler = nodeWidgetHandlers[node.comfyClass]?.[widget.name];
    if (handler) {
        handler(node, value, widget);
    }
}

function inputLogic(node, input) {
    const handler = nodeInputHandlers[node.comfyClass]?.[input.name];
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
            let inputLink = iv.link;
            let originalDescriptor = Object.getOwnPropertyDescriptor(iv, 'link');
            inputLogic(node, iv);

            Object.defineProperty(iv, 'link', {
                get() {
                    return originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(iv)
                        : inputLink;
                },
                set(newVal) {
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(iv, newVal);
                    } else {
                        inputLink = newVal;
                    }

                    inputLogic(node, iv);
                }
            });
        }
    },
});
