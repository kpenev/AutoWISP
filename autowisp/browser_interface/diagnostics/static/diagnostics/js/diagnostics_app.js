function selectSymbol(event)
{
    let marker = event.currentTarget.className.baseVal.split(" ")[1];
    let master_id = event.currentTarget.parentElement.id.split(":")[1];
    let button = document.getElementById("marker-button:" + master_id);
    button.replaceChild(event.currentTarget.cloneNode(true), button.children[0]);
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

document.addEventListener("DOMContentLoaded", function() {
    if ( !initDiagnosticsPlotting.done )
        initDiagnosticsPlotting();

    const rows = document.querySelectorAll(".diagnostic-row");
    rows.forEach(function(row) {
        row.addEventListener("click", function() {
            const diagId = this.dataset.diagnosticId;
            this.classList.toggle("active");
        });
    });
});
