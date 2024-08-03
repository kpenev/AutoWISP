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

function getPlotConfig()
{
    const markerButtons = document.getElementsByClassName("selected-marker");
    let plotConfig = {
        'datasets': {},
        'x_range': [
            document.getElementById("plot-x-min").value,
            document.getElementById("plot-x-max").value
        ],
        'y_range': [
            document.getElementById("plot-y-min").value,
            document.getElementById("plot-y-max").value
        ],
        'mag_expression': document.getElementById("mag-expression").value
    }
    for ( const button of markerButtons ) {
        let marker = button.children[0].className.baseVal.split(" ")[1];
        if ( marker != "" ) {
            let masterId = button.id.split(":")[1];
            plotConfig['datasets'][masterId] = {
                "color": document.getElementById("plot-color:" 
                                                 + 
                                                 masterId).value,
                "marker": marker,
                "min_fraction": document.getElementById("min-fraction:" 
                                                        + 
                                                        masterId).value,
                "label": document.getElementById("label:" 
                                                 + 
                                                 masterId).value,
            }
        }
    }
    return plotConfig;
}

function showNewPlot(data)
{
    let plotParent = document.getElementById("plot-parent");
    for ( child of plotParent.children )
        if ( child.tagName.toUpperCase() == "SVG" )
            plotParent.removeChild(child);

    plotParent.innerHTML = 
        data["plot_data"]
        +
        document.getElementById("plot-parent").innerHTML;

    document.getElementById("plot-x-min").value = 
        data["plot_config"]["x_range"][0];
    document.getElementById("plot-x-max").value =
        data["plot_config"]["x_range"][1];

    document.getElementById("plot-y-min").value = 
        data["plot_config"]["y_range"][0];
    document.getElementById("plot-y-max").value =
        data["plot_config"]["y_range"][1];


    setPlotSize();
}

function updatePlot()
{
    postJson(updatePlot.url, getPlotConfig())
        .then((response) => {
            console.log(response);
            return response.json();
        })
        .then((data) => {
            console.log(data);
            showNewPlot(data);
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

function startEditPlot(event)
{
    event.preventDefault();
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
    let plot = document.getElementById("plot-parent").children[0];
    plot.addEventListener("dblclick", startEditPlot);
}
