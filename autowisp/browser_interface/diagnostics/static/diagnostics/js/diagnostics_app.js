function navigateDiagnostics()
{
    const bar = document.getElementById('diag-selector-bar');
    const yDiag = document.getElementById('diagnostic-selector').value;
    const xDiag = document.getElementById('x-diagnostic-selector').value;
    if (xDiag === 'time') {
        window.location.href = bar.dataset.imageUrl.replace('YPLACEHOLDER', yDiag);
    } else {
        window.location.href = bar.dataset.diagVsDiagUrl
            .replace('XPLACEHOLDER', xDiag)
            .replace('YPLACEHOLDER', yDiag);
    }
}

function selectSymbol(event)
{
    let marker = event.currentTarget.className.baseVal.split(" ")[1];
    let master_id = event.currentTarget.parentElement.id.split(":")[1];
    let button = document.getElementById("marker-button:" + master_id);
    button.replaceChild(event.currentTarget.cloneNode(true), button.children[0]);
}

function getSelectedDatasets()
{
    const activeRows = document.querySelectorAll(".diagnostic-row.active");
    let datasets = {};
    for ( const row of activeRows ) {
        let seriesId = row.id;
        let button = document.getElementById("marker-button:" + seriesId);
        let marker = button.children[0].className.baseVal.split(" ")[1];
        if ( marker != "" ) {
            datasets[seriesId] = {
                "channel": row.getAttribute("channel"),
                "color": document.getElementById(
                    "plot-color:" + seriesId
                ).value,
                "marker": marker,
                "scale": document.getElementById(
                    "scale:" + seriesId
                ).value,
                "label": document.getElementById(
                    "label:" + seriesId
                ).value,
            };
        }
    }
    let display = document.getElementById("diagnostics-display");
    let rect = display.getBoundingClientRect();
    let legendToggle = document.getElementById("legend-toggle");
    return {
        "datasets": datasets,
        "figure_config": {
            "aspect_ratio": rect.width / rect.height,
            "show_legend": !legendToggle || !legendToggle.classList.contains("inactive"),
        },
    };
}

function showDiagnosticsPlot(data)
{
    let downloadBtn = document.getElementById("download-button");
    if (downloadBtn)
        downloadBtn.style.display = "inline";
    let display = document.getElementById("diagnostics-display");
    display.innerHTML = "";
    showSVG(data, "diagnostics-display");
    display.style.height = "";
    display.style.minHeight = "";
    display.style.maxHeight = "";
    let figure = display.children[0];
    let aspectRatio = (stripUnits(figure.getAttribute("width"))
                       / stripUnits(figure.getAttribute("height")));
    let style = getComputedStyle(display);
    let contentWidth = (display.clientWidth
                        - parseFloat(style.paddingLeft)
                        - parseFloat(style.paddingRight));
    figure.setAttribute("width", contentWidth);
    figure.setAttribute("height", contentWidth / aspectRatio);
}

function initDiagnosticsPlotting(plotURL)
{
    const plotSymbols = document.getElementsByClassName("plot-marker");
    for ( const symbol of plotSymbols ) {
        if ( symbol.parentElement.className == "dropdown-content" )
            symbol.addEventListener("click", selectSymbol);
    }

    if ( plotURL ) {
        updateFigure.url = plotURL;
        updateFigure.callback = showNewPlot;
        updateFigure.getParam = getPlotConfig;

        document.getElementById("plot-button").onclick = updateFigure;
        document.getElementById("plot-sep").addEventListener(
            "mousedown", sepDragStart
        );
        document.getElementById("plot-config-parent").addEventListener(
            "scroll", scrollConfig
        );

        updateFigure();
    }

    initDiagnosticsPlotting.done = true;
}

function initImageDiagnostics(plotURL)
{
    initDiagnosticsPlotting();

    updateFigure.url = plotURL;
    updateFigure.callback = showDiagnosticsPlot;
    updateFigure.getParam = getSelectedDatasets;

    const rows = document.querySelectorAll(".diagnostic-row");
    rows.forEach(function(row) {
        row.addEventListener("click", function() {
            this.classList.toggle("active");
            updateFigure();
        });
    });
}

document.addEventListener("DOMContentLoaded", function() {
    if ( !initDiagnosticsPlotting.done )
        initDiagnosticsPlotting();
});
