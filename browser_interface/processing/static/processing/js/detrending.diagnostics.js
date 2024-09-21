function selectSymbol(event)
{
    let marker = event.currentTarget.className.baseVal.split(" ")[1];
    let master_id = event.currentTarget.parentElement.id.split(":")[1];
    let button = document.getElementById("marker-button:" + master_id);
    button.replaceChild(event.currentTarget.cloneNode(true), button.children[0]);
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
        'mag_expression': [
            document.getElementById("mag-expression").value,
            document.getElementById("mag-label").value
        ],
        'marker_size': document.getElementById("marker-size").value,

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
                "scale": document.getElementById("scale:" 
                                                 + 
                                                 masterId).value,
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
    showSVG(data, "plot-parent");
    document.getElementById("plot-x-min").value = 
        data["plot_config"]["x_range"][0];
    document.getElementById("plot-x-max").value =
        data["plot_config"]["x_range"][1];

    document.getElementById("plot-y-min").value = 
        data["plot_config"]["y_range"][0];
    document.getElementById("plot-y-max").value =
        data["plot_config"]["y_range"][1];

    document.getElementById("mag-expression").value =
        data["plot_config"]["mag_expression"][0];

    document.getElementById("mag-label").value =
        data["plot_config"]["mag_expression"][1];

    document.getElementById("marker-size").value = 
        data["plot_config"]["marker_size"];

    setFigureSize("plot-parent");
    document.getElementById("download-button").style.display="inline";
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
    setFigureSize("plot-parent");
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

function scrollConfig(event)
{
    const markerDropdowns = document.getElementsByClassName("dropdown-content");
    for ( const entry of markerDropdowns ) {
        entry.style.top = entry.
            parentNode.
            getElementsByClassName("dropbtn selected-marker")[0].
            getBoundingClientRect().bottom + "px";
    }
}

function initDiagnosticsPlotting(plotURL) 
{
    const plotSymbols = document.getElementsByClassName("plot-marker");
    for ( const symbol of plotSymbols ) {
        if ( symbol.parentElement.className == "dropdown-content" )
            symbol.addEventListener("click", selectSymbol);
    }
    document.getElementById("plot-button").addEventListener("click", 
                                                            updateFigure);
    updateFigure.url = plotURL;
    updateFigure.callback = showNewPlot;
    updateFigure.getParam = getPlotConfig;
    document.getElementById("plot-sep").addEventListener("mousedown",
                                                         sepDragStart)
    let plot = document.getElementById("plot-parent").children[0];
    plot.addEventListener("dblclick", startEditPlot);
    document.getElementById("plot-config-parent").
        addEventListener("scroll", scrollConfig);
}
