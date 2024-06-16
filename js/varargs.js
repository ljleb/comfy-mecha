// original src: https://github.com/jags111/efficiency-nodes-comfyui
import { app } from "../../scripts/app.js";

let origVarargsInputs = null;
let origBlocksWidgets = null;

const MAX_VARARGS_MODELS = 64;  // arbitrary limit to n-models methods (open an issue if this is a problem)

const findWidgetIndexByName = (widgets, name) => {
    return widgets ? widgets.findIndex((w) => w.name === name) : null;
};

function handleMechaModelListVisibility(node, visibleCount) {
    if (origVarargsInputs === null) {
        origVarargsInputs = node.inputs;
    }

    for (let i = 0; i < node.inputs.length; ++i) {
        const input = node.inputs[i];
        const origInput = origVarargsInputs[i];

        for (const key of Object.keys(input)) {
            origInput[key] = input[key];
        }
    }

    node.inputs = Array.from(origVarargsInputs);
    node.inputs.length = visibleCount;
    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

function handleMechaHyperBlocksListVisibility(node, preset) {
    if (origBlocksWidgets == null) {
        origBlocksWidgets = node.widgets;
    }

    for (let i = 0; i < node.widgets.length; ++i) {
        const widget = node.widgets[i];
        origBlocksWidgets[findWidgetIndexByName(origBlocksWidgets, widget.name)] = widget;
    }

    if (preset === "custom") {
        node.widgets = Array.from(origBlocksWidgets);
    } else {
        node.widgets = Array.from(origBlocksWidgets);

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

function handleMechaModelListVisibilityByCount(node, widget) {
    handleMechaModelListVisibility(node, widget.value);
}

function handleMechaHyperBlocksVisibilityByPreset(node, widget) {
    handleMechaHyperBlocksListVisibility(node, widget.value);
}

function widgetLogic(node, widget) {
    const handler = nodeWidgetHandlers[node.comfyClass]?.[widget.name];
    if (handler) {
        handler(node, widget);
    }
}

const nodeWidgetHandlers = {
    "Mecha Recipe List": {
        "count": handleMechaModelListVisibilityByCount
    },
    "Blocks Mecha Hyper": {
        "preset": handleMechaHyperBlocksVisibilityByPreset
    },
};

app.registerExtension({
    name: "mecha.widgethider",
    nodeCreated(node) {
        for (const w of node.widgets || []) {
            let widgetValue = w.value;
            let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value');
            widgetLogic(node, w);

            Object.defineProperty(w, 'value', {
                get() {
                    return originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;
                },
                set(newVal) {
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else {
                        widgetValue = newVal;
                    }

                    widgetLogic(node, w);
                }
            });
        }
    }
});
