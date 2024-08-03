// original src: https://github.com/jags111/efficiency-nodes-comfyui
import { app } from "../../scripts/app.js";

let customDefInputNames = {};

const MAX_VARARGS_MODELS = 64;  // arbitrary limit to n-models methods (open an issue if this is a problem)

const findWidgetIndexByName = (widgets, name) => {
    return widgets ? widgets.findIndex((w) => w.name === name) : null;
};

function handleMechaModelListVisibility(node, visibleCount) {
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
    node.inputs.length = visibleCount;
    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

function handleMechaHyperBlocksListVisibility(node, preset) {
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
    },
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
	    if (nodeData.output[0] === "MECHA_RECIPE" && nodeData.input && nodeData.input.required) {
	        for (const ik in nodeData.input.required) {
	            if (nodeData.input.required[ik][0] === "MECHA_RECIPE" || nodeData.input.required[ik][0] === "MECHA_RECIPE_LIST") {
	                if (!(nodeData.display_name in customDefInputNames)) {
	                    customDefInputNames[nodeData.display_name] = [];
	                }
                    customDefInputNames[nodeData.display_name].push(ik);
	            }
	        }
        }
	},
	loadedGraphNode(node, app) {
		for (const iv of node.inputs || []) {
		    if (iv.type === "MECHA_RECIPE" || iv.type === "MECHA_RECIPE_LIST") {
		        for (const ik of customDefInputNames[node.title] || []) {
		            if (ik.split(" ")[0] === iv.name.split(" ")[0]) {
		                iv.name = ik;
		            }
		        }
		    }
		}
	},
});
