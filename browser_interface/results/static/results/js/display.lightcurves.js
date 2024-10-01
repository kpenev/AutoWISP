var configURLs

//Check if a given boundary should be triggered by the given event location.
function triggerBoundary(event, box)
{
    let side;
    if ( event.offsetX > box.left && event.offsetX < box.left + 20 )
        side = "left";
    else if ( event.offsetX > box.right - 20  && event.offsetX < box.right )
        side = "right";
    else if ( event.offsetY > box.top && event.offsetY < box.top + 20 )
        side = "top";
    else if ( event.offsetY > box.bottom - 20 && event.offsetY < box.bottom )
        side = "bottom";
    else
        return null;
    
    if ( side == "left" || side == "right" ) 
        return {
            side: side, 
            fraction: (event.offsetY - box.top) / (box.bottom - box.top)
        };
    else
        return {
            side: side, 
            fraction: (event.offsetX - box.left) / (box.right - box.left)
        };
}

function triggerSubPlot(event, box)
{
    if (
        event.offsetX > box.left
        && event.offsetX < box.right
        && event.offsetY > box.top
        && event.offsetY < box.bottom
       ) {
        let plotHighlight = document.getElementById("plot-highlight");
        plotHighlight.style.display = "inline";
        plotHighlight.style.left = box.left + "px";
        plotHighlight.style.top = box.top + "px";
        plotHighlight.style.width = box.right - box.left + "px";
        plotHighlight.style.height = box.bottom - box.top + "px";
        plotHighlight.onmouseleave = (event) => {plotHighlight.style.display = "none";}
        return true;
    }
    return false;
}

function highlightPlotBoundary(which, box, figureBounds)
{
    let parentBounds = document
        .getElementById("figure-parent")
        .getBoundingClientRect();
    let plotSplit = document.getElementById("plot-split");
    let rem = parseFloat(getComputedStyle(plotSplit).fontSize);

    for ( side of ["left", "right", "top", "bottom"] ) {
        plotSplit.style.removeProperty(side);
        plotSplit.classList.remove(side);
    }

    plotSplit.style.removeProperty("width");
    plotSplit.style.removeProperty("height");

    plotSplit.classList.add(which);
    plotSplit.style.display = "inline";
    if ( which == "left" || which == "right" ) {
        plotSplit.style.top = (box.top 
                               + figureBounds.top 
                               - parentBounds.top 
                               + "px");
        plotSplit.style.height = (box.bottom 
                                  - box.top 
                                  - 1.5 * rem 
                                  + "px");

        if ( which == "left" ) 
            plotSplit.style.left = (box.left
                                    + figureBounds.left
                                    - parentBounds.left
                                    + "px");
        else
            plotSplit.style.right = (parentBounds.right 
                                     + figureBounds.width 
                                     - box.right 
                                     - figureBounds.right 
                                     + "px");

        splitBounds = plotSplit.getBoundingClientRect();
    } else {
        plotSplit.style.left = (box.left 
                                + figureBounds.left 
                                - parentBounds.left
                                + "px");
        plotSplit.style.width = (box.right 
                                 - box.left 
                                 - 1.5 * rem 
                                 + "px");
        if ( which == "top" )
            plotSplit.style.top = (box.top
                                   + figureBounds.top
                                   - parentBounds.top
                                   + "px");
        else
            plotSplit.style.bottom = (parentBounds.bottom 
                                      - figureBounds.bottom 
                                      + figureBounds.height 
                                      - box.bottom 
                                      + "px");

    }
    figureMouseOver.action = which;

    plotSplit.addEventListener("mouseleave", unhighlightPlotBoundary);
}

function cleanSplits(removeUnapplied)
{
    if ( removeUnapplied ) {
        getPlottingConfig.unappliedSplits = {};
        elements = document.querySelectorAll(".unapplied");
    } else
        elements = document.querySelectorAll(".temporary");

    elements.forEach(e => e.parentNode.removeChild(e));
}

function unhighlightPlotBoundary()
{
    cleanSplits(false);
    if ( typeof figureMouseOver.action !== "undefined" ) {
        let plotSplit = document.getElementById("plot-split");
        plotSplit.classList.remove(figureMouseOver.action);
        plotSplit.style.display = "none";
    }
    document
        .getElementById("plot-split")
        .removeEventListener("mouseleave", unhighlightPlotBoundary);

}

function addSplits(splitBoundary, box, plotId, splitRange, splitCount)
{
    splitCount -= 2;
    showExtraSplit(splitBoundary, box, plotId, splitCount);
    document
        .querySelectorAll(".temporary")
        .forEach(e => e.classList.remove("temporary"));
    event.preventDefault();
    document.getElementById("plot-split").onclick = null;
    figure = document.getElementById("figure-parent").children[0];
    if ( !(plotId in getPlottingConfig.unappliedSplits) )
        getPlottingConfig.unappliedSplits[plotId] = {};

    const currentSplits = 
        getPlottingConfig.unappliedSplits[plotId][splitBoundary.side];
    const newSplits = new Array(splitCount + 1);
    newSplits.fill((splitRange[1] - splitRange[0]) / (splitCount + 1))

    if ( typeof currentSplits === "undefined" )
        getPlottingConfig.unappliedSplits[plotId][splitBoundary.side] = 
            newSplits;
    else {
        let splicePos = 0;
        for ( let right = 0; 
              right < splitRange[0]; 
              right += currentSplits[splicePos] )
            splicePos++;
        currentSplits.splice(splicePos, 1, ...newSplits);
    }

    if ( splitBoundary.side == "left" )
        getPlottingConfig.unappliedSplits[plotId]["right"] = 
            getPlottingConfig.unappliedSplits[plotId]["left"];
    else if ( splitBoundary.side == "right" )
        getPlottingConfig.unappliedSplits[plotId]["left"] = 
            getPlottingConfig.unappliedSplits[plotId]["right"];
    if ( splitBoundary.side == "top" )
        getPlottingConfig.unappliedSplits[plotId]["bottom"] = 
            getPlottingConfig.unappliedSplits[plotId]["top"];
    else if ( splitBoundary.side == "bottom" )
        getPlottingConfig.unappliedSplits[plotId]["top"] = 
            getPlottingConfig.unappliedSplits[plotId]["bottom"];
}

function showExtraSplit(splitBoundary, box, plotId, splitCount)
{
    cleanSplits(false);
    let figureParent = document.getElementById("figure-parent");
    let figure = figureParent.children[0];
    let unappliedSplits = getPlottingConfig.unappliedSplits;
    let splitRange = [0.0, 1.0];
    if ( 
        plotId in unappliedSplits
        &&
        splitBoundary.side in unappliedSplits[plotId]
       ) {
        for ( const splitSize of unappliedSplits[plotId][splitBoundary.side] ){
            if ( splitRange[0] + splitSize > splitBoundary.fraction ) {
                splitRange[1] = splitRange[0] + splitSize;
                break;
            }
            splitRange[0] += splitSize;
        }
    }
    for ( let splitInd = 1; splitInd <= splitCount; splitInd++) {
        let splitFraction = (
                             (splitInd * splitRange[1]
                             + 
                             (splitCount + 1 - splitInd) * splitRange[0])
                            ) / (splitCount + 1); 

        let newSplit = document.createElement("hr");
        newSplit.style.position = "absolute";
        newSplit.classList.add("split", "unapplied", "temporary")

        if ( splitBoundary.side == "top" || splitBoundary.side == "bottom" ) {
            newSplit.classList.add("vertical");
            newSplit.style.top = box.top + "px";
            newSplit.style.height = box.bottom - box.top + "px";
            newSplit.style.left = ((1.0 - splitFraction) * box.left
                                   +
                                   splitFraction * box.right
                                   +
                                   "px");
        } else {
            newSplit.classList.add("horizontal");
            newSplit.style.left = box.left + "px";
            newSplit.style.width = box.right - box.left + "px";
            newSplit.style.top = (
                                  (1.0 - splitFraction) * box.top
                                   +
                                   splitFraction * box.bottom
                                  +
                                  "px"
                                 );
        }
        figureParent.appendChild(newSplit);
    }

    document.onkeyup = cleanSplits.bind(null, false);
    plotSplit = document.getElementById("plot-split");
    plotSplit.onclick = function(event) {
        if ( event.shiftKey && splitCount > 1 ) 
            showExtraSplit(splitBoundary,
                           box,
                           plotId,
                           splitCount - 1);

        else if ( !event.shiftKey )
            showExtraSplit(splitBoundary,
                           box,
                           plotId,
                           splitCount + 1);
    }
    plotSplit.ondblclick = addSplits.bind(null,
                                          splitBoundary,
                                          box,
                                          plotId,
                                          splitRange,
                                          splitCount);
    document.onkeydown = null;
}

//Return the plot ID, box and boundary where this event occurred (each could be
//null)
function identifySubPlot(event)
{
    let figureParent = document.getElementById("figure-parent");
    let figure = figureParent.children[0];
    let figureBounds = figure.getBoundingClientRect();

    let shifted_event = {
        offsetX: event.clientX - figureBounds.left,
        offsetY: event.clientY - figureBounds.top
    };

    let box;
    let activeBoundary;
    for ( plotId of Object.keys(figure.boundaries) ) {
        box = { ...figure.boundaries[plotId] };
        box.left *= figureBounds.width;
        box.right *= figureBounds.width;
        box.top *= figureBounds.height;
        box.bottom *= figureBounds.height;

        if ( triggerSubPlot(shifted_event, box) ) {
            return [plotId, box, triggerBoundary(shifted_event, box)]
        }
    }
    return [null, null, null];
}

function figureMouseOver(event)
{
    const [plotId, box, activeBoundary] = identifySubPlot(event);
    if ( activeBoundary !== null ) {
        let figureBounds = document
            .getElementById("figure-parent")
            .children[0]
            .getBoundingClientRect();
        highlightPlotBoundary(activeBoundary.side, 
                              box, 
                              figureBounds);
        if ( event.ctrlKey ) {
            showExtraSplit(activeBoundary, 
                           box, 
                           plotId, 
                           1);
        } else {
            document.onkeydown = showExtraSplit.bind(null,
                                                     activeBoundary, 
                                                     box, 
                                                     plotId,
                                                     1);
        }
        return;
    }
}

function getPlottingConfig()
{
    const result = {applySplits: getPlottingConfig.unappliedSplits};

    if ( typeof getPlottingConfig.plotId === "undefined" )
        return result;
    result[getPlottingConfig.mode] = {plotId: getPlottingConfig.plotId};
    for ( const element of document.getElementsByClassName("param") ) {
        result[getPlottingConfig.mode][element.id] = element.value;
    }
    return result;
}

function showConfig(url, parentId, onSuccess)
{
    const request = new XMLHttpRequest();
    request.open("GET", url);
    request.send();
    request.onload = () => {
        let configParent = document.getElementById(parentId);
        configParent.innerHTML = request.responseText;
        configParent.parentNode.style.display = "inline-flex";
        configParent.style.display = "inline";
        if ( typeof onSuccess !== "undefined" )
            onSuccess();
    }
}

function showEditPlot(event)
{
    const [plotId, box, activeBoundary] = identifySubPlot(event);
    if ( plotId !== null && activeBoundary === null ) {
        showConfig(
                   configURLs.subplot.slice(0, -1) + plotId, 
                   "config-parent",
                   () => {
                       document
                           .getElementById("select-model")
                           .onchange = changeModel;

                       document
                           .getElementById("add-expression")
                           .onclick = addNewExpressionConfig;

                       getPlottingConfig.mode = "subplot";
                   }
                  );
        getPlottingConfig.plotId = plotId;
    }
}

function showEditRc(event)
{
    const request = new XMLHttpRequest();
    request.open("GET", configURLs.rcParams);
    request.send();
    request.onreadystatechange = function() {
        document.getElementById("config-parent").innerHTML = 
            request.responseText;
    }
}

function changeModel()
{
    modelSelect = document.getElementById("select-model");
    if ( modelSelect.value == "" )
        document.getElementById("define-model").style.display = "none";
    else
        showConfig(configURLs.editModel.slice(0, -1) + modelSelect.value,
                   "define-model");
}

function showNewFigure(data)
{
    cleanSplits(true);
    boundaries = showSVG(data, "figure-parent")["boundaries"];
    let figureParent = document.getElementById("figure-parent");
    let figure = figureParent.children[0];
    figure.boundaries = boundaries;
    setFigureSize("figure-parent");
    figureParent.addEventListener("mousemove", figureMouseOver);
    figureParent.onclick = showEditPlot;
}

function addNewExpressionConfig()
{
    let expressionsParent = document
        .getElementById("lc-expressions")
        .getElementsByTagName("tbody")[0];
    let lastRow = expressionsParent.lastElementChild.previousElementSibling;
    let newRow = lastRow.cloneNode(true);
    for ( input of newRow.getElementsByTagName("input")) {
        let lastDashPos = input.id.lastIndexOf('-');
        let counter = parseInt(input.id.slice(lastDashPos + 1)) + 1;
        input.id = input.id.slice(0, lastDashPos + 1) + counter;
        input.value = "";
    }
    lastRow.after(newRow);
}

function initLightcurveDisplay(urls)
{
    updateFigure.url = urls.update;
    delete urls.update;
    configURLs = urls;
    updateFigure.callback = showNewFigure;
    updateFigure.getParam = getPlottingConfig;
    getPlottingConfig.unappliedSplits = {};
    document
        .getElementById("rcParams")
        .onclick = (() => showConfig(urls.rcParams,
                                     "config-parent",
                                     () => {
                                         getPlottingConfig.mode = "rcParams";
                                     }));
    document.getElementById("apply").onclick = updateFigure;
    updateFigure();
}
