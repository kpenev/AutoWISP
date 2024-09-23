//Check if a given boundary should be triggered by the given event location.
function triggerBoundary(event, box)
{
    let side;
    if ( 
        event.offsetX > box.left 
        && event.offsetX < box.left + 10 
        && event.offsetY > box.top
        && event.offsetY < box.bottom
       )
        side = "left";
    else if ( 
        event.offsetX > box.right - 10 
        && event.offsetX < box.right 
        && event.offsetY > box.top
        && event.offsetY < box.bottom
       )
        side = "right";
    else if ( 
        event.offsetY > box.top 
        && event.offsetY < box.top + 10 
        && event.offsetX > box.left
        && event.offsetX < box.right
       )
        side = "top";
    else if ( 
        event.offsetY > box.bottom - 10 
        && event.offsetY < box.bottom 
        && event.offsetX > box.left
        && event.offsetX < box.right
       )
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

function highlightPlotBoundary(which, box, figureBounds)
{
    let parentBounds = document
        .getElementById("figure-parent")
        .getBoundingClientRect();
    let plotSplit = document.getElementById("plot-split");
    let rem = parseFloat(getComputedStyle(plotSplit).fontSize);

    plotSplit.style.removeProperty("left");
    plotSplit.style.removeProperty("right");
    plotSplit.style.removeProperty("top");
    plotSplit.style.removeProperty("bottom");
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

function removeAllTemporary()
{
    document
        .querySelectorAll(".temporary")
        .forEach(e => e.parentNode.removeChild(e));
}

function unhighlightPlotBoundary()
{
    removeAllTemporary();
    if ( typeof figureMouseOver.action !== "undefined" ) {
        let plotSplit = document.getElementById("plot-split");
        plotSplit.classList.remove(figureMouseOver.action);
        plotSplit.style.display = "none";
    }
    document
        .getElementById("plot-split")
        .removeEventListener("mouseleave", unhighlightPlotBoundary);

}

function addSplits(splitBoundary, box, plotId, splitCount)
{
    showExtraSplit(splitBoundary, box, plotId, splitCount - 2);
    document
        .querySelectorAll(".temporary")
        .forEach(e => e.classList.remove("temporary"));
    event.preventDefault();
    document.getElementById("plot-split").onclick = null;
    figure = document.getElementById("figure-parent").children[0];
    if ( !(plotId in figure.unappliedSplits) )
        figure.unappliedSplits[plotId] = {};
    figure.unappliedSplits[plotId][splitBoundary.side] = 
        Array.from({length: splitCount},
                   function(_, i) {
                       return (i + 1) / splitCount;
                   });
    if ( splitBoundary.side == "left" )
        figure.unappliedSplits[plotId]["right"] = 
            figure.unappliedSplits[plotId]["left"];
    else if ( splitBoundary.side == "right" )
        figure.unappliedSplits[plotId]["left"] = 
            figure.unappliedSplits[plotId]["right"];
    if ( splitBoundary.side == "top" )
        figure.unappliedSplits[plotId]["bottom"] = 
            figure.unappliedSplits[plotId]["top"];
    else if ( splitBoundary.side == "bottom" )
        figure.unappliedSplits[plotId]["top"] = 
            figure.unappliedSplits[plotId]["bottom"];
}

function showExtraSplit(splitBoundary, box, plotId, splitCount)
{
    removeAllTemporary();
    let figureParent = document.getElementById("figure-parent");
    let figure = figureParent.children[0];
    let unappliedSplits = figure.unappliedSplits;
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
        newSplit.classList.add("temporary");

        if ( splitBoundary.side == "top" || splitBoundary.side == "bottom" ) {
            newSplit.style.top = box.top + "px";
            newSplit.style.height = box.bottom - box.top + "px";
            newSplit.style.width = "1px";
            newSplit.style.left = ((1.0 - splitFraction) * box.left
                                   +
                                   splitFraction * box.right
                                   +
                                   "px");
        } else {
            newSplit.style.left = box.left + "px";
            newSplit.style.width = box.right - box.left + "px";
            newSplit.style.height = "1px";
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

    document.onkeyup = removeAllTemporary;
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
    plotSplit.ondblclick = function() {addSplits(splitBoundary,
                                                 box,
                                                 plotId,
                                                 splitCount);}
    document.onkeydown = null;
}

function figureMouseOver(event)
{
    let figureParent = document.getElementById("figure-parent");
    let figure = figureParent.children[0];
    let figureBounds = figure.getBoundingClientRect();

    let box;
    let activeBoundary;
    Object.keys(figure.boundaries).forEach(function(plotId) {
        box = { ...figure.boundaries[plotId] };
        box.left *= figureBounds.width;
        box.right *= figureBounds.width;
        box.top *= figureBounds.height;
        box.bottom *= figureBounds.height;
        activeBoundary = triggerBoundary(event, box);
        if ( activeBoundary !== null ) {
            highlightPlotBoundary(activeBoundary.side, box, figureBounds);
            if ( event.ctrlKey ) {
                showExtraSplit(activeBoundary, box, plotId, 1);
            } else {
                document.onkeydown = (function(event) {
                    showExtraSplit(activeBoundary, 
                                   box, 
                                   plotId,
                                   1);
                });
            }
            return;
        }
    })
}

function showNewFigure(data)
{
    boundaries = showSVG(data, "figure-parent")["boundaries"];
    let figure = document.getElementById("figure-parent").children[0];
    figure.boundaries = boundaries;
    figure.unappliedSplits = {};
    setFigureSize("figure-parent");
    figure.addEventListener("mousemove", figureMouseOver);
}

function initLightcurveDisplay(updateURL)
{
    updateFigure.url = updateURL;
    updateFigure.callback = showNewFigure;
    updateFigure();
}
