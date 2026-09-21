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
    // Every row, not only the drawn ones: the server answers a rebinding
    // out of the row's own posted state, and a row switched off is rebound
    // like any other.
    const rows = document.querySelectorAll(".diagnostic-row");
    let datasets = {};
    for ( const row of rows ) {
        let seriesId = row.id;
        let button = document.getElementById("marker-button:" + seriesId);
        let marker = button.children[0].className.baseVal.split(" ")[1];
        let color = document.getElementById("plot-color:" + seriesId);
        let label = document.getElementById("label:" + seriesId);
        datasets[seriesId] = {
            "selected": row.classList.contains("active"),
            // The observing session and image type, opaque here: the row
            // posts back what the dropdown says and the server takes it
            // apart, as it does the row id.
            "pair": row.querySelector(".pair-select").value,
            "channels": getRowChannels(row),
            "color": color.value,
            "marker": marker,
            "scale": document.getElementById(
                "scale:" + seriesId
            ).value,
            "label": label.value,
            // Whether each still holds what it was rendered with, which
            // is what tells the server it may replace them with the
            // defaults of a new binding -- before the figure is drawn, so
            // the plot and the table never disagree about a row that has
            // just been rebound. The same test applyTableResponse makes
            // when the answer comes back, so both mean the same fields.
            "automatic_color": color.value === color.dataset.default,
            "automatic_label": label.value === label.dataset.default,
        };
    }
    let display = document.getElementById("diagnostics-display");
    let rect = display.getBoundingClientRect();
    let legendToggle = document.getElementById("legend-toggle");

    // Set by whatever was just done to a row -- a dropdown rebinding it,
    // or `+` asking for a copy of it -- and cleared here so that the next
    // redraw, a colour or a marker or a row switched on, does not ask for
    // the same thing all over again.
    const bind = getSelectedDatasets.bind;
    const add = getSelectedDatasets.add;
    getSelectedDatasets.bind = null;
    getSelectedDatasets.add = null;

    return {
        "datasets": datasets,
        "bind": bind,
        "add": add,
        "figure_config": {
            "aspect_ratio": rect.width / rect.height,
            "show_legend": !legendToggle || !legendToggle.classList.contains("inactive"),
        },
    };
}

function applyTableResponse(data)
{
    // The copy `+` asked for, inserted directly below the row it was
    // taken from. Directly below holds under any sort, the sort library
    // not re-sorting when a row appears.
    if ( data.added_row ) {
        const source = document.getElementById(data.after);
        if ( source ) {
            source.insertAdjacentHTML("afterend", data.added_row);
            wireDiagnosticRow(source.nextElementSibling);
            refreshRemoveButtons();
        }
    }

    // What a rebound row earns: its count, the channel cells its pair
    // offers, that session's start and end, and the defaults that follow
    // from what it now binds.
    if ( !data.bind )
        return;
    const row = document.getElementById(data.bind);
    if ( !row )
        return;

    const count = row.querySelector(".series-count");
    if ( count && data.count !== undefined )
        count.textContent = data.count;

    for ( const [selector, value] of [[".session-start", data.start],
                                      [".session-end", data.end]] ) {
        const cell = row.querySelector(selector);
        if ( cell && value !== undefined )
            cell.textContent = value;
    }

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

    if ( data.slot_cells !== undefined && count ) {
        // Replaced rather than edited: the number of columns never
        // changes, but which channels each may offer does, and the server
        // renders them so that the page keeps one renderer for a cell.
        // They sit between the session times and the count, which is what
        // the count cell is used to find.
        for ( const cell of row.querySelectorAll(".slot-cell") )
            cell.remove();
        count.insertAdjacentHTML("beforebegin", data.slot_cells);
        wireSlotCells(row);
    }
}

function showDiagnosticsPlot(data)
{
    applyTableResponse(data);
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

function onRowChange(event)
{
    // One handler for both of a row's dropdowns, which ask the same
    // question of the server: redraw, and tell me what this row binds now.
    const row = event.target.closest(".diagnostic-row");

    // The pair always earns an answer, because which channels each column
    // may offer depends on the session and image type, so its cells are
    // re-rendered whatever they hold. A channel earns one only once every
    // column is set: before that the row names no data to bind or count.
    if ( event.target.matches(".pair-select") || isRowBound(row) )
        getSelectedDatasets.bind = row.id;

    // With nothing to ask and nothing drawn, nothing has changed that
    // anyone can see.
    if ( getSelectedDatasets.bind || row.classList.contains("active") )
        updateFigure();
}

function onAddRow(event)
{
    // Asked for on the redraw the new row needs anyway, rather than in a
    // request of its own: the payload already carries the clicked row's
    // state and every row id on the page, which is all the server needs
    // to build the copy and give it an id of its own.
    getSelectedDatasets.add = event.target.closest(".diagnostic-row").id;
    updateFigure();
}

function onRemoveRow(event)
{
    // Nothing to ask the server, the table being the only place this row
    // exists: it goes, and the figure is redrawn without it. Nothing is
    // lost that `+` and the dropdowns cannot build again, which is why
    // this asks for no confirmation.
    event.target.closest(".diagnostic-row").remove();
    refreshRemoveButtons();
    updateFigure();
}

function refreshRemoveButtons()
{
    // The page needs a row to draw anything at all, so the last one
    // cannot be removed. Said with a disabled button rather than by
    // refusing the click, so that it is visible before it is tried.
    const rows = document.querySelectorAll(".diagnostic-row");
    for ( const row of rows )
        row.querySelector(".remove-row").disabled = rows.length < 2;
}

function stopClick(event)
{
    event.stopPropagation();
}

function refreshSortKey(event)
{
    // A dropdown cell sorts by `data-sort`, the text of the option chosen
    // in it, since the cell's own text is every option run together. One
    // delegated listener rather than one per dropdown, so that a cell the
    // server replaces goes on sorting without being wired again.
    const select = event.target;
    if ( !select.matches("select") )
        return;
    const cell = select.closest("td");
    if ( cell && select.selectedIndex >= 0 )
        cell.dataset.sort = select.options[select.selectedIndex].text.trim();
}

function wireSlotCells(row)
{
    // Called again whenever the server replaces these cells, which it does
    // every time the row's pair changes.
    for ( const select of row.querySelectorAll(".slot-cell select") ) {
        select.addEventListener("click", stopClick);
        select.addEventListener("change", onRowChange);
    }
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
    for ( const control of
          row.querySelectorAll("input, select, button, .dropdown") )
        control.addEventListener("click", stopClick);

    row.querySelector(".add-row").addEventListener("click", onAddRow);
    row.querySelector(".remove-row").addEventListener("click", onRemoveRow);

    // A colour, a scale or a label changes only how a series looks, so
    // there is nothing to ask the server and the figure is simply
    // redrawn. On `change` rather than `input`: dragging through a colour
    // picker then redraws once it is settled rather than at every shade
    // on the way, and a label redraws when it is finished rather than per
    // keystroke.
    for ( const input of row.querySelectorAll("input") )
        input.addEventListener("change", updateFigure);

    // The row's marker menu, wired here so that a row built after page
    // load gets a working one too. Rows the page-load pass already covered
    // are unaffected: adding the same listener to the same element twice
    // has no effect.
    for ( const symbol of row.querySelectorAll(".plot-marker") )
        if ( symbol.parentElement.className == "dropdown-content" )
            symbol.addEventListener("click", selectSymbol);

    row.querySelector(".pair-select").addEventListener("change", onRowChange);
    wireSlotCells(row);
}

function initImageDiagnostics(plotURL)
{
    initDiagnosticsPlotting();

    updateFigure.url = plotURL;
    updateFigure.callback = showDiagnosticsPlot;
    updateFigure.getParam = getSelectedDatasets;

    document.getElementById("diagnostics-table-parent").addEventListener(
        "change", refreshSortKey
    );
    document.querySelectorAll(".diagnostic-row").forEach(wireDiagnosticRow);
    refreshRemoveButtons();

    // A row arrives drawn, so the figure is asked for at once rather than
    // waiting for a first click that no longer has to happen.
    updateFigure();
}

document.addEventListener("DOMContentLoaded", function() {
    if ( !initDiagnosticsPlotting.done )
        initDiagnosticsPlotting();
});
