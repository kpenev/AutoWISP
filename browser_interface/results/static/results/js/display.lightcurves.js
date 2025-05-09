var configURLs;
var plotCurves;

//Check if a given boundary should be triggered by the given event location.
function triggerBoundary(event, box) {
    let side;
    if (event.offsetX > box.left && event.offsetX < box.left + 20)
        side = "left";
    else if (event.offsetX > box.right - 20 && event.offsetX < box.right)
        side = "right";
    else if (event.offsetY > box.top && event.offsetY < box.top + 20)
        side = "top";
    else if (event.offsetY > box.bottom - 20 && event.offsetY < box.bottom)
        side = "bottom";
    else return null;

    if (side == "left" || side == "right")
        return {
            side: side,
            fraction: (event.offsetY - box.top) / (box.bottom - box.top),
        };
    else
        return {
            side: side,
            fraction: (event.offsetX - box.left) / (box.right - box.left),
        };
}

function triggerSubPlot(event, box) {
    if (
        event.offsetX > box.left &&
        event.offsetX < box.right &&
        event.offsetY > box.top &&
        event.offsetY < box.bottom
    ) {
        const plotHighlight = document.getElementById("plot-highlight");
        plotHighlight.style.display = "inline";
        plotHighlight.style.left = box.left + "px";
        plotHighlight.style.top = box.top + "px";
        plotHighlight.style.width = box.right - box.left + "px";
        plotHighlight.style.height = box.bottom - box.top + "px";
        plotHighlight.onmouseleave = (event) => {
            plotHighlight.style.display = "none";
        };
        return true;
    }
    return false;
}

function highlightPlotBoundary(which, box, figureBounds) {
    const parentBounds = document
        .getElementById("figure-parent")
        .getBoundingClientRect();
    const plotSplit = document.getElementById("plot-split");
    const rem = parseFloat(getComputedStyle(plotSplit).fontSize);

    for (side of ["left", "right", "top", "bottom"]) {
        plotSplit.style.removeProperty(side);
        plotSplit.classList.remove(side);
    }

    plotSplit.style.removeProperty("width");
    plotSplit.style.removeProperty("height");

    plotSplit.classList.add(which);
    plotSplit.style.display = "inline";
    if (which == "left" || which == "right") {
        plotSplit.style.top =
            box.top + figureBounds.top - parentBounds.top + "px";
        plotSplit.style.height = box.bottom - box.top - 1.5 * rem + "px";

        if (which == "left")
            plotSplit.style.left =
                box.left + figureBounds.left - parentBounds.left + "px";
        else
            plotSplit.style.right =
                parentBounds.right +
                figureBounds.width -
                box.right -
                figureBounds.right +
                "px";

        splitBounds = plotSplit.getBoundingClientRect();
    } else {
        plotSplit.style.left =
            box.left + figureBounds.left - parentBounds.left + "px";
        plotSplit.style.width = box.right - box.left - 1.5 * rem + "px";
        if (which == "top")
            plotSplit.style.top =
                box.top + figureBounds.top - parentBounds.top + "px";
        else
            plotSplit.style.bottom =
                parentBounds.bottom -
                figureBounds.bottom +
                figureBounds.height -
                box.bottom +
                "px";
    }
    figureMouseOver.action = which;

    plotSplit.addEventListener("mouseleave", unhighlightPlotBoundary);
}

function cleanSplits(removeUnapplied) {
    if (removeUnapplied) {
        getPlottingConfig.unappliedSplits = {};
        elements = document.querySelectorAll(".unapplied");
    } else elements = document.querySelectorAll(".temporary");

    elements.forEach((e) => e.parentNode.removeChild(e));
}

function unhighlightPlotBoundary() {
    cleanSplits(false);
    if (typeof figureMouseOver.action !== "undefined") {
        const plotSplit = document.getElementById("plot-split");
        plotSplit.classList.remove(figureMouseOver.action);
        plotSplit.style.display = "none";
    }
    document
        .getElementById("plot-split")
        .removeEventListener("mouseleave", unhighlightPlotBoundary);
}

function addSplits(splitBoundary, box, plotId, splitRange, splitCount) {
    splitCount -= 2;
    showExtraSplit(splitBoundary, box, plotId, splitCount);
    document
        .querySelectorAll(".temporary")
        .forEach((e) => e.classList.remove("temporary"));
    event.preventDefault();
    document.getElementById("plot-split").onclick = null;
    figure = document.getElementById("figure-parent").children[0];
    if (!(plotId in getPlottingConfig.unappliedSplits))
        getPlottingConfig.unappliedSplits[plotId] = {};

    const currentSplits =
        getPlottingConfig.unappliedSplits[plotId][splitBoundary.side];
    const newSplits = new Array(splitCount + 1);
    newSplits.fill((splitRange[1] - splitRange[0]) / (splitCount + 1));

    if (typeof currentSplits === "undefined")
        getPlottingConfig.unappliedSplits[plotId][splitBoundary.side] =
            newSplits;
    else {
        let splicePos = 0;
        for (
            let right = 0;
            right < splitRange[0];
            right += currentSplits[splicePos]
        )
            splicePos++;
        currentSplits.splice(splicePos, 1, ...newSplits);
    }

    if (splitBoundary.side == "left")
        getPlottingConfig.unappliedSplits[plotId]["right"] =
            getPlottingConfig.unappliedSplits[plotId]["left"];
    else if (splitBoundary.side == "right")
        getPlottingConfig.unappliedSplits[plotId]["left"] =
            getPlottingConfig.unappliedSplits[plotId]["right"];
    if (splitBoundary.side == "top")
        getPlottingConfig.unappliedSplits[plotId]["bottom"] =
            getPlottingConfig.unappliedSplits[plotId]["top"];
    else if (splitBoundary.side == "bottom")
        getPlottingConfig.unappliedSplits[plotId]["top"] =
            getPlottingConfig.unappliedSplits[plotId]["bottom"];
}

function showExtraSplit(splitBoundary, box, plotId, splitCount) {
    cleanSplits(false);
    const figureParent = document.getElementById("figure-parent");
    const figure = figureParent.children[0];
    const unappliedSplits = getPlottingConfig.unappliedSplits;
    let splitRange = [0.0, 1.0];
    if (
        plotId in unappliedSplits &&
        splitBoundary.side in unappliedSplits[plotId]
    ) {
        for (const splitSize of unappliedSplits[plotId][splitBoundary.side]) {
            if (splitRange[0] + splitSize > splitBoundary.fraction) {
                splitRange[1] = splitRange[0] + splitSize;
                break;
            }
            splitRange[0] += splitSize;
        }
    }
    for (let splitInd = 1; splitInd <= splitCount; splitInd++) {
        const splitFraction =
            (splitInd * splitRange[1] +
                (splitCount + 1 - splitInd) * splitRange[0]) /
            (splitCount + 1);

        const newSplit = document.createElement("hr");
        newSplit.style.position = "absolute";
        newSplit.classList.add("split", "unapplied", "temporary");

        if (splitBoundary.side == "top" || splitBoundary.side == "bottom") {
            newSplit.classList.add("vertical");
            newSplit.style.top = box.top + "px";
            newSplit.style.height = box.bottom - box.top + "px";
            newSplit.style.left =
                (1.0 - splitFraction) * box.left +
                splitFraction * box.right +
                "px";
        } else {
            newSplit.classList.add("horizontal");
            newSplit.style.left = box.left + "px";
            newSplit.style.width = box.right - box.left + "px";
            newSplit.style.top =
                (1.0 - splitFraction) * box.top +
                splitFraction * box.bottom +
                "px";
        }
        figureParent.appendChild(newSplit);
    }

    document.onkeyup = cleanSplits.bind(null, false);
    plotSplit = document.getElementById("plot-split");
    plotSplit.onclick = function (event) {
        if (event.shiftKey && splitCount > 1)
            showExtraSplit(splitBoundary, box, plotId, splitCount - 1);
        else if (!event.shiftKey)
            showExtraSplit(splitBoundary, box, plotId, splitCount + 1);
    };
    plotSplit.ondblclick = addSplits.bind(
        null,
        splitBoundary,
        box,
        plotId,
        splitRange,
        splitCount
    );
    document.onkeydown = null;
}

//Return the plot ID, box and boundary where this event occurred (each could be
//null)
function identifySubPlot(event) {
    const figureParent = document.getElementById("figure-parent");
    const figure = figureParent.children[0];
    const figureBounds = figure.getBoundingClientRect();

    const shifted_event = {
        offsetX: event.clientX - figureBounds.left,
        offsetY: event.clientY - figureBounds.top,
    };

    let box;
    for (plotId of Object.keys(figure.boundaries)) {
        box = { ...figure.boundaries[plotId] };
        box.left *= figureBounds.width;
        box.right *= figureBounds.width;
        box.top *= figureBounds.height;
        box.bottom *= figureBounds.height;

        if (triggerSubPlot(shifted_event, box)) {
            return [plotId, box, triggerBoundary(shifted_event, box)];
        }
    }
    return [null, null, null];
}

function figureMouseOver(event) {
    const [plotId, box, activeBoundary] = identifySubPlot(event);
    if (activeBoundary !== null) {
        const figureBounds = document
            .getElementById("figure-parent")
            .children[0].getBoundingClientRect();
        highlightPlotBoundary(activeBoundary.side, box, figureBounds);
        if (event.ctrlKey) {
            showExtraSplit(activeBoundary, box, plotId, 1);
        } else {
            document.onkeydown = showExtraSplit.bind(
                null,
                activeBoundary,
                box,
                plotId,
                1
            );
        }
        return;
    }
}

function getPlottingConfig() {
    const result = { applySplits: getPlottingConfig.unappliedSplits };

    if (typeof getPlottingConfig.plotId === "undefined") return result;
    const subplotConfig = { plot_id: getPlottingConfig.plotId }
    if (getPlottingConfig.mode == "subplot") {
        plotCurves.saveLoadSelection();
        subplotConfig["data_select"] = plotCurves.configuredCurves;

        for (const decoration of [
            "x-label", 
            "y-label", 
            "title", 
            "xmin", 
            "xmax",
            "ymin", 
            "ymax"
        ])
            subplotConfig[decoration.replaceAll("-", "_")] = document
                .getElementById(decoration)
                .value
        for (const idComponent of ["star-id-type", "star-id"])
            subplotConfig[idComponent.replaceAll("-", "_")] = document
                .getElementById(idComponent)
                .value

        const selectedModel = document.getElementById("select-model").value;
        if (selectedModel) {
            subplotConfig["model"] = getModelParameters();
            subplotConfig["model"]["type"] = selectedModel;
        }
    }

    result[getPlottingConfig.mode] = subplotConfig;
    return result;
}

function showConfig(url, parentId, onSuccess) {
    const request = new XMLHttpRequest();
    request.open("GET", url);
    request.send();
    request.onload = () => {
        const configParent = document.getElementById(parentId);
        configParent.innerHTML = request.responseText;
        configParent.parentNode.style.display = "inline-flex";
        configParent.style.display = "inline";
        if (typeof onSuccess !== "undefined") onSuccess();
    };
}

function showEditPlot(event) {
    const [plotId, box, activeBoundary] = identifySubPlot(event);
    if (plotId !== null && activeBoundary === null) {
        showConfig(
            configURLs.subplot.slice(0, -1) + plotId,
            "config-parent",
            () => {
                document.getElementById("select-model").onchange = changeModel;

                for (const param_group of [
                    "substitution",
                    "lc-expression",
                    "find-best",
                ])
                    document.getElementById("add-" + param_group).onclick =
                        () => addNewParam(param_group);
                document.getElementById("lc-expressions").onchange =
                    handleLCParamChange;

                const lcDataSelect = JSON.parse(
                    document.getElementById("lc-data-select").textContent
                );
                plotCurves = new plotCurvesType(lcDataSelect);
                getPlottingConfig.mode = "subplot";
            }
        );
        getPlottingConfig.plotId = plotId;
    }
}

function showEditRc(event) {
    const request = new XMLHttpRequest();
    request.open("GET", configURLs.rcParams);
    request.send();
    request.onreadystatechange = function () {
        document.getElementById("config-parent").innerHTML =
            request.responseText;
    };
}

function getModelParameters() {
    const modelDefine = document.getElementById("define-model");
    const model = {};
    for (element of modelDefine.getElementsByClassName("param")) {
        model[element.id.substring(6)] = element.value;
    }
    return model;
}

function changeModel() {
    const modelSelect = document.getElementById("select-model");
    const modelDefine = document.getElementById("define-model");
    if (modelSelect.value == "") {
        modelDefine.style.display = "none";
        changeModel.stashed[changeModel.currentModel] = getModelParameters();
    } else {
        showConfig(
            configURLs.editModel.slice(0, -3) + modelSelect.value + "/0",
            "define-model",
            () => {
                changeModel.currentModel = modelSelect.value;
                if (modelSelect.value in changeModel.stashed) {
                    model = changeModel.stashed[changeModel.currentModel];
                    for (element of modelDefine.getElementsByClassName(
                        "param"
                    )) {
                        element.value = model[element.id.substring(6)];
                    }
                }
            }
        );
    }

    for (const updateId of ["x", "y", "match_by"]) {
        updateElement = document.getElementById(updateId);
        if (
            modelSelect.value == "" 
            && 
            updateElement.lastElementChild.value == 'best_model'
        )
            updateElement.removeChild(updateElement.lastElementChild);
        else if (updateElement.lastElementChild.value != 'best_model') {
            const option = document.createElement("option");
            option.value = "best_model";
            option.textContent = "best_model";
            updateElement.appendChild(option);

        }
    }
}

function showNewFigure(data) {
    cleanSplits(true);
    boundaries = showSVG(data, "figure-parent")["boundaries"];
    const figureParent = document.getElementById("figure-parent");
    const figure = figureParent.children[0];
    figure.boundaries = boundaries;
    setFigureSize("figure-parent");
    figureParent.addEventListener("mousemove", figureMouseOver);
    figureParent.onclick = showEditPlot;
}

//Allow user to edit the curves in a given data selection to display.
class plotCurvesType {
    //Initialize the object.
    constructor(configuredCurves) {
        this.configuredCurves = configuredCurves;
        this.selectionInd = 0;
        this.curveInd = 0;
        this.elements = {
            define: document.getElementById("define-curve"),
            nextSelection: document.getElementById("next-selection"),
            prevSelection: document.getElementById("previous-selection"),
            nextCurve: document.getElementById("next-curve"),
            prevCurve: document.getElementById("previous-curve"),
        };
        this.elements.prevSelection.onclick = () => this.switchSelection(-1);
        this.elements.nextSelection.onclick = () => this.switchSelection(1);
        this.elements.prevCurve.onclick = () => this.switchCurve(-1);
        this.elements.nextCurve.onclick = () => this.switchCurve(1);
        this.fixVisual();
        const plotSymbols = document.getElementsByClassName("plot-marker");
        for ( const symbol of plotSymbols ) {
            if ( symbol.parentElement.id == "marker-option" )
                symbol.addEventListener("click", selectSymbol);
        }

    }

    //Deep copy the first curve in the current selection.
    cloneCurve(selectionInd, curveInd) {
        if (typeof selectionInd === "undefined")
            selectionInd = this.selectionInd;
        if (typeof curveInd === "undefined") curveInd = 0;
        return JSON.parse(
            JSON.stringify(
                this.configuredCurves[selectionInd].plot_config[curveInd]
            )
        );
    }

    //Return a deep copy the first selection but with only the first curve.
    createNewSelection() {
        const result = {};
        for (const key in this.configuredCurves[0]) {
            if (key == "plot_config") result[key] = [this.cloneCurve(0, 0)];
            else
                result[key] = JSON.parse(
                    JSON.stringify(this.configuredCurves[0][key])
                );
        }
        return result;
    }

    //Visually indicate whether first or last curve/data selection is selected.
    fixVisual() {
        this.saveLoadSelection(true);
        if (this.curveInd == 0) {
            this.elements.prevCurve.classList.add("lcars-melrose-bg");
            const indicatorClassList =
                this.elements.prevCurve.firstElementChild.classList;

            indicatorClassList.remove("fa-chevron-left");
            indicatorClassList.add("fa-plus");
        } else {
            this.elements.prevCurve.classList.remove("lcars-melrose-bg");
            const indicatorClassList =
                this.elements.prevCurve.firstElementChild.classList;

            indicatorClassList.remove("fa-plus");
            indicatorClassList.add("fa-chevron-left");
        }
        if (
            this.curveInd ==
            this.configuredCurves[this.selectionInd].plot_config.length - 1
        ) {
            this.elements.nextCurve.classList.add("lcars-melrose-bg");
            const indicatorClassList =
                this.elements.nextCurve.firstElementChild.classList;

            indicatorClassList.remove("fa-chevron-right");
            indicatorClassList.add("fa-plus");
        } else {
            this.elements.nextCurve.classList.remove("lcars-melrose-bg");
            const indicatorClassList =
                this.elements.nextCurve.firstElementChild.classList;

            indicatorClassList.remove("fa-plus");
            indicatorClassList.add("fa-chevron-right");
        }
        if (this.selectionInd == 0) {
            this.elements.prevSelection.classList.add("lcars-melrose-bg");
            const indicatorClassList =
                this.elements.prevSelection.firstElementChild.classList;

            indicatorClassList.remove("fa-chevron-left");
            indicatorClassList.add("fa-plus");
        } else {
            this.elements.prevSelection.classList.remove("lcars-melrose-bg");
            const indicatorClassList =
                this.elements.prevSelection.firstElementChild.classList;

            indicatorClassList.remove("fa-plus");
            indicatorClassList.add("fa-chevron-left");
        }
        if (this.selectionInd == this.configuredCurves.length - 1) {
            this.elements.nextSelection.classList.add("lcars-melrose-bg");
            const indicatorClassList =
                this.elements.nextSelection.firstElementChild.classList;

            indicatorClassList.remove("fa-chevron-right");
            indicatorClassList.add("fa-plus");
        } else {
            this.elements.nextSelection.classList.remove("lcars-melrose-bg");
            const indicatorClassList =
                this.elements.nextSelection.firstElementChild.classList;

            indicatorClassList.remove("fa-plus");
            indicatorClassList.add("fa-chevron-right");
        }
    }

    //Save edits to the currently selected plot curve
    saveLoadCurve(load) {
        const curveConfig = this.configuredCurves[
            this.selectionInd
        ].plot_config[
            this.curveInd
        ];

        for (const element of this.elements.define.getElementsByClassName(
            "param"
        )) {
            let config;
            if (element.classList.contains('kwarg')) {
                config = curveConfig.plot_kwargs;
            } else {
                config = curveConfig;
            }
            if (element.id == "marker") {
                let marker = element.children[0];
                if (load)
                    marker.className.baseVal = "plot-marker " +  config.marker;
                else
                    config.marker = marker.className.baseVal.split(" ")[1];
            } else if (load) {
                element.value = config[element.id]
            } else {
                config[element.id] = element.value;
            }
        }
    }

    //Save edits to the currently selected selection, including the curve
    saveLoadSelection(load) {
        this.saveLoadCurve(load);
        const thisSelection = this.configuredCurves[this.selectionInd];
        const paramGroupTargets = {
            "substitution": "lc_substitutions",
            "find-best": "find_best",
            "lc-expression": "expressions"
        }
        for (const groupName in paramGroupTargets) {
            const target = thisSelection[paramGroupTargets[groupName]];
            const selectorStr = "[id^='" + groupName + "-key-']";
            for (
                const keyElement 
                of 
                document
                .getElementById(groupName + "s")
                .querySelectorAll(selectorStr)
            ) {
                const valueElement = document.getElementById(
                    keyElement.id.replace("-key-", "-value-")
                );
                if (load)
                    valueElement.value = target[keyElement.value];
                else {
                    if (keyElement.value == "magfit_iteration") 
                        target[keyElement.value] = parseInt(valueElement.value);
                    else
                        target[keyElement.value] = valueElement.value;
                }
            }
        }
        if (load) {
            document.getElementById("minimize").value = thisSelection["minimize"];
            document.getElementById("include-apphot").checked = 
                thisSelection["photometry_modes"].includes("apphot");
            document.getElementById("include-shapefit").checked =
                thisSelection["photometry_modes"].includes("shapefit");
            document.getElementById("points-selection").value = 
                thisSelection["selection"];
            const model = thisSelection['model'];
            if (model) {
                document.getElementById("select-model").value = model.type;
                changeModel.stashed[model.type] = JSON.parse(
                    JSON.stringify(model.kwargs)
                );
                changeModel.stashed[model.type]["shift"] =
                    Boolean(changeModel.stashed[model.type]["shift_to"]);
                changeModel.stashed[model.type]["quantity"] = model["quantity"];

                changeModel();
                /*
                document.getElementById("model-quantity").value = 
                    model["quantity"];
                document.getElementById("model-shift").checked = 
                    model["shift_to"];
                for ( const param in model.kwargs )
                    document.getElementById["model-" + param] = model[param];
                    */
            } else {
                document.getElementById("select-model").value = "";
            }
        } else {
            thisSelection["minimize"] = document.getElementById("minimize").value;
            thisSelection["photometry_modes"] = [];
            if (document.getElementById("include-apphot").checked)
                thisSelection["photometry_modes"].push("apphot");
            if (document.getElementById("include-shapefit").checked)
                thisSelection["photometry_modes"].push("shapefit");
            thisSelection["selection"] = 
                document.getElementById("points-selection").value;

            const modelType = document.getElementById("select-model").value;
            if (modelType) {
                const model = {
                    "type": modelType,
                    "quantity": document.getElementById("model-quantity").value,
                    "shift_to": document.getElementById("model-shift").checked,
                    "kwargs": {}
                };
                for( 
                    const element 
                    of 
                    document
                    .getElementById("model-params")
                    .getElementsByTagName("input")
                ) {
                    model["kwargs"][element.id.substring(6)] = element.value;
                }
                thisSelection['model'] = model;
            } else {
                thisSelection['model'] = null;
            }
        }
    }

    //Switch the curve being configured, saving anything edited.
    switchCurve(step) {
        if (step) {
            this.saveLoadCurve();

            this.curveInd += step;
            if (this.curveInd < 0) {
                this.configuredCurves[this.selectionInd].plot_config.splice(
                    0,
                    0,
                    this.cloneCurve()
                );
                this.curveInd++;
            } else if (
                this.curveInd ==
                this.configuredCurves[this.selectionInd].plot_config.length
            )
                this.configuredCurves[this.selectionInd].plot_config.push(
                    this.cloneCurve()
                );
        }
        this.fixVisual();
    }

    //Switch the data selection being, saving any edits.
    switchSelection(step) {
        if (step) {
            this.saveLoadSelection();
            this.selectionInd += step;
            if (this.selectionInd < 0) {
                this.configuredCurves.splice(0, 0, this.createNewSelection());
                this.selectionInd++;
            } else if (this.selectionInd == this.configuredCurves.length)
                this.configuredCurves.push(this.createNewSelection());
            if (step) this.curveInd = 0;
            this.switchCurve(0);
        } else 
            this.fixVisual();
    }
}

function addNewParam(param_group) {
    const expressionsParent = document
        .getElementById(param_group + "s")
        .getElementsByTagName("tbody")[0];
    const lastRow = expressionsParent.lastElementChild.previousElementSibling;
    const newRow = lastRow.cloneNode(true);
    for (input of newRow.getElementsByTagName("input")) {
        const lastDashPos = input.id.lastIndexOf("-");
        const counter = parseInt(input.id.slice(lastDashPos + 1)) + 1;
        input.id = input.id.slice(0, lastDashPos + 1) + counter;
        input.value = "";
    }
    lastRow.after(newRow);
}

function handleLCParamChange(event) {
    plotCurves.saveLoadSelection();
    if (!event.target.id.startsWith("lc-expression-key"))
        return;
    const changeInd = parseInt(event.target.id.substring(18));
    for (const updateId of ["x", "y", "match_by"]) {
        updateElement = document.getElementById(updateId);
        let numExpressions = updateElement.childElementCount;
        let addElement;
        if (updateElement.lastElementChild.value == "best_model") {
            numExpressions--;
            addElement = (
                (opt) => updateElement.insertBefore(
                    opt,
                    updateElement.lastElementChild
                )
            );
        } else 
            addElement = updateElement.appendChild;
        if (changeInd < numExpressions ) {
            updateElement.children[changeInd].value = event.target.value;
            updateElement.children[changeInd].textContent =
                event.target.value;
            return;
        }
        for (
            let ind = numExpressions;
            ind <= changeInd;
            ind++
        ) {
            const option = document.createElement("option");
            option.value = document
                .getElementById("lc-expression-key-" + ind)
                .value;
            option.textContent = option.value;
            addElement(option);
        }
    }
}

function selectSymbol(event)
{
    let markerButton = document.getElementById("marker");
    markerButton.replaceChild(event.currentTarget.cloneNode(true), 
                              markerButton.children[0]);
}

function initLightcurveDisplay(urls) {
    updateFigure.url = urls.update;
    delete urls.update;
    configURLs = urls;
    updateFigure.callback = showNewFigure;
    updateFigure.getParam = getPlottingConfig;
    changeModel.stashed = {};
    getPlottingConfig.unappliedSplits = {};
    document.getElementById("rcParams").onclick = () =>
        showConfig(urls.rcParams, "config-parent", () => {
            getPlottingConfig.mode = "rcParams";
        });
    document.getElementById("apply").onclick = updateFigure;
    updateFigure();
}
