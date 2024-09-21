//Check if a given boundary should be triggered by the given event location.
function triggerBoundary(event, box)
{
    if ( 
        event.offsetX > box.left 
        && event.offsetX < box.left + 10 
        && event.offsetY > box.top
        && event.offsetY < box.bottom
       )
        return "left";
    if ( 
        event.offsetX > box.right - 10 
        && event.offsetX < box.right 
        && event.offsetY > box.top
        && event.offsetY < box.bottom
       )
        return "right";
    if ( 
        event.offsetY > box.top 
        && event.offsetY < box.top + 10 
        && event.offsetX > box.left
        && event.offsetX < box.right
       )
        return "top";
    if ( 
        event.offsetY > box.bottom - 10 
        && event.offsetY < box.bottom 
        && event.offsetX > box.left
        && event.offsetX < box.right
       )
        return "bottom";
}

function highlightPlotBoundary(which, boundaries, figureBounds)
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
        plotSplit.style.top = (boundaries.top 
                               + figureBounds.top 
                               - parentBounds.top 
                               + "px");
        plotSplit.style.height = (boundaries.bottom 
                                  - boundaries.top 
                                  - 1.5 * rem 
                                  + "px");

        if ( which == "left" ) 
            plotSplit.style.left = (boundaries.left
                                    + figureBounds.left
                                    - parentBounds.left
                                    + "px");
        else
            plotSplit.style.right = (parentBounds.right 
                                     + figureBounds.width 
                                     - boundaries.right 
                                     - figureBounds.right 
                                     + "px");

        splitBounds = plotSplit.getBoundingClientRect();
    } else {
        plotSplit.style.left = (boundaries.left 
                                + figureBounds.left 
                                - parentBounds.left
                                + "px");
        plotSplit.style.width = (boundaries.right 
                                 - boundaries.left 
                                 - 1.5 * rem 
                                 + "px");
        if ( which == "top" )
            plotSplit.style.top = (boundaries.top
                                   + figureBounds.top
                                   - parentBounds.top
                                   + "px");
        else
            plotSplit.style.bottom = (parentBounds.bottom 
                                      - figureBounds.bottom 
                                      + figureBounds.height 
                                      - boundaries.bottom 
                                      + "px");

    }
    figureMouseOver.action = which;

    document
        .getElementById("plot-split")
        .addEventListener("mouseleave", unhighlightPlotBoundary);
}

function unhighlightPlotBoundary()
{
    if ( typeof figureMouseOver.action !== "undefined" ) {
        let plotSplit = document.getElementById("plot-split");
        plotSplit.classList.remove(figureMouseOver.action);
        plotSplit.style.display = "none";
    }
    document
        .getElementById("plot-split")
        .removeEventListener("mouseleave", unhighlightPlotBoundary);

}

function figureMouseOver(event)
{

    let figureParent = document.getElementById("figure-parent");
    let figure = figureParent.children[0];
    let figureBounds = figure.getBoundingClientRect();

    let boundaries;
    let activeBoundary;
    Object.keys(figureMouseOver.boundaries).forEach(function(plot_id) {
        boundaries = { ...figureMouseOver.boundaries[plot_id] };
        boundaries.left *= figureBounds.width;
        boundaries.right *= figureBounds.width;
        boundaries.top *= figureBounds.height;
        boundaries.bottom *= figureBounds.height;
        activeBoundary = triggerBoundary(event, boundaries);
        if ( typeof activeBoundary !== "undefined" ) {
            highlightPlotBoundary(activeBoundary, boundaries, figureBounds);
            return;
        }
    })
}

function showNewFigure(data)
{
    figureMouseOver.boundaries = showSVG(data, "figure-parent")["boundaries"];
    setFigureSize("figure-parent");
    let figure = document.getElementById("figure-parent").children[0];
    figure.addEventListener("mousemove", figureMouseOver);
}

function initLightcurveDisplay(updateURL)
{
    updateFigure.url = updateURL;
    updateFigure.callback = showNewFigure;
    updateFigure();
}
