// original src: https://github.com/jags111/efficiency-nodes-comfyui
import { app } from "../../scripts/app.js";

let origInputs = null;

const MAX_VARARGS_MODELS = 64;  // arbitrary limit to n-models methods (open an issue if this is a problem)

const findWidgetByName = (node, name) => {
    return node.inputs ? node.inputs.find((w) => w.name === name) : null;
};

function handleMechaModelListVisibility(node, visibleCount) {
    if (origInputs === null) {
        origInputs = node.inputs;
    }

    for (let i = 0; i < node.inputs.length; ++i) {
        const input = node.inputs[i];
        const origInput = origInputs[i];

        for (const key of Object.keys(input)) {
            origInput[key] = input[key];
        }
    }

    node.inputs = Array.from(origInputs);
    node.inputs.length = visibleCount;
    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

function handleMechaModelListVisibilityByCount(node, widget) {
    handleMechaModelListVisibility(node, widget.value);
}

const nodeWidgetHandlers = {
    "Mecha Recipe List": {
        "count": handleMechaModelListVisibilityByCount
    },
};

function widgetLogic(node, widget) {
    const handler = nodeWidgetHandlers[node.comfyClass]?.[widget.name];
    if (handler) {
        handler(node, widget);
    }
}

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
