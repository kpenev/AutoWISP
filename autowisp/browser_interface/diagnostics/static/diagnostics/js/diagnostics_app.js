function navigateDiagnostics()
{
    const bar = document.getElementById('diag-selector-bar');
    const yDiag = document.getElementById('diagnostic-selector').value;
    const xDiag = document.getElementById('x-diagnostic-selector').value;
    window.location.href = bar.dataset.diagnosticsUrl
        .replace('XPLACEHOLDER', xDiag)
        .replace('YPLACEHOLDER', yDiag);
}

function selectSymbol(event)
{
    let marker = event.currentTarget.className.baseVal.split(" ")[1];
    let master_id = event.currentTarget.parentElement.id.split(":")[1];
    let button = document.getElementById("marker-button:" + master_id);
    button.replaceChild(event.currentTarget.cloneNode(true), button.children[0]);

    // A series table draws on every edit, so the figure has to catch up
    // with the style just chosen.  The detrending page, whose rows are not
    // .diagnostic-row, redraws when its Plot button is pressed instead, and
    // must not be made to redraw per marker.
    if ( event.currentTarget.closest(".diagnostic-row") )
        updateFigure();
}

function getRowChannels(row)
{
    // A fixed channel is text rather than a dropdown, since there is
    // nothing to choose; either way the cell says what the row binds.
    return Array.from(row.querySelectorAll(".slot-cell")).map(
        (cell) => cell.dataset.channel
                  ?? (cell.querySelector("select") || {}).value
                  ?? ""
    );
}

function isRowBound(row)
{
    const channels = getRowChannels(row);
    return channels.every((channel) => channel !== "");
}

function getSelectedDatasets()
{
    // Every row, not only the drawn ones: the server decides from the
    // whole table whether a completed binding has earned a spare row.
    const rows = document.querySelectorAll(".diagnostic-row");
    let datasets = {};
    for ( const row of rows ) {
        let seriesId = row.id;
        let button = document.getElementById("marker-button:" + seriesId);
        let marker = button.children[0].className.baseVal.split(" ")[1];
        datasets[seriesId] = {
            "selected": row.classList.contains("active"),
            "channels": getRowChannels(row),
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
    let display = document.getElementById("diagnostics-display");
    let rect = display.getBoundingClientRect();
    let legendToggle = document.getElementById("legend-toggle");

    // Set by the dropdown that just completed a row, and cleared here so
    // that the next redraw -- a colour, a marker, a row switched on -- does
    // not ask for a count and a spare row all over again.
    const bind = getSelectedDatasets.bind;
    getSelectedDatasets.bind = null;

    return {
        "datasets": datasets,
        "bind": bind,
        "figure_config": {
            "aspect_ratio": rect.width / rect.height,
            "show_legend": !legendToggle || !legendToggle.classList.contains("inactive"),
        },
    };
}

function applyBinding(data)
{
    // What a completed row earns: its own count, the defaults that follow
    // from the channels it now names, and -- the first time -- a fresh
    // spare below it, so a second binding of the same series can be built.
    if ( !data.bind )
        return;
    const row = document.getElementById(data.bind);
    if ( !row )
        return;

    const count = row.querySelector(".series-count");
    if ( count && data.count !== undefined )
        count.textContent = data.count;

    // Only where the field still holds what it was rendered with: a colour
    // or a label the user chose is theirs, and must survive a rebinding.
    for ( const [prefix, value] of [["plot-color", data.color],
                                    ["label", data.label]] ) {
        const input = document.getElementById(prefix + ":" + data.bind);
        if ( input && value !== undefined
             && input.value === input.dataset.default ) {
            input.value = value;
            input.dataset.default = value;
        }
    }

    if ( data.spare_row ) {
        row.insertAdjacentHTML("afterend", data.spare_row);
        wireAppendedRow(row.nextElementSibling);
    }
}

function showDiagnosticsPlot(data)
{
    applyBinding(data);
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

function onSlotChange(event)
{
    const row = event.target.closest(".diagnostic-row");

    if ( !isRowBound(row) ) {
        // Nothing to fetch, count or draw until every channel is chosen,
        // so a partly bound row asks the server nothing -- unless it was
        // drawn a moment ago, and this change has just undrawn it.
        if ( row.classList.contains("active") )
            updateFigure();
        return;
    }

    getSelectedDatasets.bind = row.id;
    updateFigure();
}

function wireDiagnosticRow(row)
{
    row.addEventListener("click", function() {
        this.classList.toggle("active");
        updateFigure();
    });

    // The row's own listener fires for clicks on its descendants, so
    // editing a row would otherwise toggle it: choosing a marker or a
    // channel, or picking a colour, would undraw the series rather than
    // redraw it in what was just chosen.
    for ( const control of row.querySelectorAll("input, select, .dropdown") )
        control.addEventListener("click", (event) => event.stopPropagation());

    for ( const select of row.querySelectorAll(".slot-select") )
        select.addEventListener("change", onSlotChange);
}

function wireAppendedRow(row)
{
    wireDiagnosticRow(row);

    // initDiagnosticsPlotting wired the markers of every row the page was
    // rendered with; one appended afterwards has to be caught here.
    for ( const symbol of row.querySelectorAll(".plot-marker") )
        if ( symbol.parentElement.className == "dropdown-content" )
            symbol.addEventListener("click", selectSymbol);
}

function initImageDiagnostics(plotURL)
{
    initDiagnosticsPlotting();

    updateFigure.url = plotURL;
    updateFigure.callback = showDiagnosticsPlot;
    updateFigure.getParam = getSelectedDatasets;

    document.querySelectorAll(".diagnostic-row").forEach(wireDiagnosticRow);
}

document.addEventListener("DOMContentLoaded", function() {
    if ( !initDiagnosticsPlotting.done )
        initDiagnosticsPlotting();
});
