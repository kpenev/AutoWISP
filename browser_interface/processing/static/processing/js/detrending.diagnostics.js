function selectSymbol(event)
{
    let marker = event.currentTarget.className.baseVal.split(" ")[1];
    let master_id = event.currentTarget.parentElement.id.split(":")[1];

    let button = document.getElementById("marker-button:" + master_id);
    button.replaceChild(event.currentTarget.cloneNode(true), button.children[0]);
}

function stripUnits(quantity)
{
    while ( isNaN(Number(quantity)) ) 
        quantity = quantity.slice(0, -1);
    return Number(quantity);
}

function setPlotSize()
{
    let fullRect = document
        .getElementById("active-area")
        .getBoundingClientRect();
    let plotParent = document.getElementById("plot-parent");
    let plot = plotParent.children[0];
    let width = plot.getAttribute("width");
    let aspectRatio = (stripUnits(plot.getAttribute("width"))
                       / 
                       stripUnits(plot.getAttribute("height")));
    let parentBoundingRect = plotParent.getBoundingClientRect();
    maxHeight = (fullRect.top
                 +
                 fullRect.height
                 -
                 parentBoundingRect.top)
    plotParent.style.height = maxHeight + "px";
    plotParent.style.minHeight = maxHeight + "px";
    plotParent.style.maxHeight = maxHeight + "px";
    plot.setAttribute("height", 
                      Math.min(maxHeight, 
                               parentBoundingRect.width 
                               / 
                               aspectRatio ));
    plot.setAttribute("width",
                      Math.min(parentBoundingRect.width,
                               maxHeight * aspectRatio));
}

function updatePlot()
{
    const markerButtons = document.getElementsByClassName("selected-marker");
    let plotConfig = {}
    for ( const button of markerButtons ) {
        let marker = button.children[0].className.baseVal.split(" ")[1];
        if ( marker != "" ) {
            let masterId = button.id.split(":")[1];
            plotConfig[masterId] = {
                "color": document.getElementById("plotColor:" + masterId).value,
                "marker": marker
            }
        }
    }
    postJson(updatePlot.url, plotConfig)
        .then((response) => {
            console.log(response);
            return response.json();
        })
        .then((data) => {
            console.log(data);
            document.getElementById("plot-parent").innerHTML = 
                data["plot_data"];
            setPlotSize();
        })
        .catch(function(error) {
            alert("Updating plot failed: " + error);
        });

}

function moveSep(event)
{
    event.preventDefault();
    let config = document.getElementById("plot-config-parent");
    let configRect = config.getBoundingClientRect();
    let height = event.clientY - configRect.top;
    config.style.height = height + "px";
    config.style.minHeight = height + "px";
    config.style.maxHeight = height + "px";
    setPlotSize();
}

function sepDragEnd(event)
{
    event.preventDefault();
    let container = document.getElementsByClassName("lcars-app-container")[0];
    container.removeEventListener("mousemove", moveSep);
    container.removeEventListener("mouseup", sepDragEnd);
}

function sepDragStart(event)
{
    event.preventDefault();
    let container = document.getElementsByClassName("lcars-app-container")[0];
    container.addEventListener("mousemove", moveSep);
    container.addEventListener("mouseup", sepDragEnd);
}

function initDiagnosticsPlotting(plotURL) 
{
    const plotSymbols = document.getElementsByClassName("plot-marker");
    for ( const symbol of plotSymbols ) {
        if ( symbol.parentElement.className == "dropdown-content" )
            symbol.addEventListener("click", selectSymbol);
    }
    document.getElementById("plot-button").addEventListener("click", 
                                                            updatePlot);
    updatePlot.url = plotURL;
    document.getElementById("plot-sep").addEventListener("mousedown",
                                                         sepDragStart)
}
