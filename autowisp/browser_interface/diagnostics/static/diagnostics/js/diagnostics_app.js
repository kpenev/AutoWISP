function sectionQuantities()
{
    // In page order, which is the order the URL lists them, the order
    // they took their markers in, and the order of the legend.
    return Array.from(
        document.querySelectorAll(".diagnostics-section")
    ).map((section) => section.dataset.quantity);
}

function pageUrl(xQuantity)
{
    const bar = document.getElementById("diag-selector-bar");
    return bar.dataset.diagnosticsUrl
        .replace("XPLACEHOLDER", xQuantity)
        .replace("YPLACEHOLDER", sectionQuantities().join(","));
}

function refreshPageUrl()
{
    // So the page can be bookmarked or reloaded as it is seen, sections
    // and all. replaceState rather than pushState: Back should leave the
    // page, not undo the additions one at a time.
    history.replaceState(null, "", pageUrl(currentXQuantity()));
}

function currentXQuantity()
{
    return document.getElementById("x-diagnostic-selector").value;
}

function navigateDiagnostics()
{
    // Every section's channel columns depend on the x, so changing it
    // rebuilds the page rather than patching it -- carrying the sections
    // across, since those are what the user asked to look at. What is
    // lost is the row state, which is what changing x has always cost.
    window.location.href = pageUrl(currentXQuantity());
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
    const sectionMarker = getSelectedDatasets.sectionMarker;
    getSelectedDatasets.bind = null;
    getSelectedDatasets.add = null;
    getSelectedDatasets.sectionMarker = null;

    // Every change to what is drawn -- a row switched off, a channel
    // chosen, a row removed -- asks for a redraw, so this is the one
    // place that sees all of them.
    refreshDrawnCounts();

    return {
        "datasets": datasets,
        "bind": bind,
        "add": add,
        "section_marker": sectionMarker,
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
            // The row arrived after the counts were last taken, and it is
            // drawn by the figure this same response carries.
            refreshDrawnCounts();
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

async function addOrJumpToSection()
{
    // One control for both, because a user picking a quantity wants to
    // look at it and does not much care whether it is already there.
    const selector = document.getElementById("diagnostic-selector");
    const quantity = selector.value;

    // Back to the placeholder, so that picking the same quantity a
    // second time is still a change the selector reports.
    selector.value = "";
    if ( !quantity )
        return;

    let section = document.getElementById("section:" + quantity);
    if ( !section ) {
        const bar = document.getElementById("diag-selector-bar");
        const taken = sectionMarkers().join(",");
        const response = await fetch(
            bar.dataset.sectionUrl.replace("YPLACEHOLDER", quantity)
            + "?taken=" + encodeURIComponent(taken)
        );

        // Only a section may be inserted here. A quantity the server
        // cannot build one for -- one naming nothing that resolves --
        // comes back as something else entirely, and pasting that into
        // the table strews a second copy of the whole page across this
        // one.
        //
        // `redirected` is the test rather than `ok`, because a failure
        // does not arrive as a failing status: the error middleware
        // records the error, queues a message naming it and sends the
        // browser back where it came from, which fetch follows without
        // complaint and reports as a perfectly good 200. Following it
        // ourselves is what puts that message in front of the user.
        if ( response.redirected ) {
            window.location.href = response.url;
            return;
        }
        if ( !response.ok ) {
            alert("Could not add " + quantity + ": " + response.status);
            return;
        }

        document.getElementById("diagnostics-table-parent")
                .insertAdjacentHTML("beforeend", await response.text());

        section = document.getElementById("section:" + quantity);
        wireSection(section);
        section.querySelectorAll(".diagnostic-row").forEach(wireDiagnosticRow);
        refreshRemoveButtons();
        refreshDrawnCounts();
        refreshPageUrl();

        // Only where the new section brings something to draw. Its row is
        // bound already where each of its channel columns has a single
        // channel recorded; on a colour camera it waits to be bound and
        // the figure is exactly as it was, so redrawing would rebuild it
        // to look the same.
        if ( Array.from(section.querySelectorAll(".diagnostic-row"))
                  .some(isRowBound) )
            updateFigure();
    }

    section.classList.remove("collapsed");
    section.scrollIntoView({block: "start"});
}

function sectionMarkers()
{
    return Array.from(
        document.querySelectorAll(".diagnostics-section")
    ).map((section) => section.dataset.marker);
}

function onToggleSection(event)
{
    // Collapsed, a section still shows its header -- what it draws, and
    // how much of the plot came from it -- so what is hidden is only the
    // rows, which is what takes the room. The caret follows the class in
    // CSS, so nothing here has to keep a glyph in step.
    event.currentTarget
         .closest(".diagnostics-section")
         .classList.toggle("collapsed");
}

function onRemoveSection(event)
{
    removeSection(event.target.closest(".diagnostics-section"));
    updateFigure();
}

function removeSection(section)
{
    section.remove();
    refreshRemoveButtons();
    refreshDrawnCounts();
    refreshPageUrl();
}

function refreshDrawnCounts()
{
    // What each section contributes to the plot: its rows that are both
    // switched on and bound, an unbound row naming no data to draw.
    for ( const section of document.querySelectorAll(".diagnostics-section") ) {
        const drawn = Array.from(
            section.querySelectorAll(".diagnostic-row")
        ).filter(
            (row) => row.classList.contains("active") && isRowBound(row)
        );
        section.querySelector(".section-drawn").textContent = drawn.length;
    }
}

function wireSection(section)
{
    // One listener, on the whole section, rather than one per part that
    // ought to respond: the bracket and the header both exist to say
    // where a section begins and neither does anything else, and so does
    // any space beside them. What must *not* collapse the section is its
    // rows, so the body stops the click before it reaches here.
    section.addEventListener("click", onToggleSection);
    section.querySelector(".section-body").addEventListener("click", stopClick);

    // Inside the header, so its click would collapse the section on the
    // way out without this.
    const remove = section.querySelector(".remove-section");
    remove.addEventListener("click", stopClick);
    remove.addEventListener("click", onRemoveSection);
}

function onAddRow(event)
{
    // Asked for on the redraw the new row needs anyway, rather than in a
    // request of its own: the payload already carries the clicked row's
    // state and every row id on the page, which is all the server needs
    // to build the copy and give it an id of its own.
    const row = event.target.closest(".diagnostic-row");
    const section = row.closest(".diagnostics-section");

    getSelectedDatasets.add = row.id;
    // The copy starts with the section's marker rather than with the
    // marker of the row it was copied from, which may have been set by
    // hand. The detrending page has no sections and sends none, and the
    // server then keeps the source's marker.
    getSelectedDatasets.sectionMarker = section ? section.dataset.marker : null;
    updateFigure();
}

function onRemoveRow(event)
{
    // Nothing to ask the server, the table being the only place this row
    // exists: it goes, and the figure is redrawn without it. Nothing is
    // lost that `+` and the dropdowns cannot build again, which is why
    // this asks for no confirmation.
    const row = event.target.closest(".diagnostic-row");
    const section = row.closest(".diagnostics-section");

    row.remove();

    // A section with no rows left draws nothing and offers nothing, so it
    // goes with its last row -- unless it is the only section, the page
    // needing a quantity to name in its URL.
    if ( section
         && !section.querySelector(".diagnostic-row")
         && document.querySelectorAll(".diagnostics-section").length > 1 )
        removeSection(section);
    else
        refreshRemoveButtons();

    updateFigure();
}

function refreshRemoveButtons()
{
    // The page needs a row to draw anything at all, and a section to name
    // in its URL, so the last of each cannot be removed. Said with a
    // disabled button rather than by refusing the click, so that it is
    // visible before it is tried.
    const rows = document.querySelectorAll(".diagnostic-row");
    for ( const row of rows )
        row.querySelector(".remove-row").disabled = rows.length < 2;

    // Empty on the detrending page, which loads this file but has no
    // sections, so this does nothing there.
    const sections = document.querySelectorAll(".diagnostics-section");
    for ( const section of sections )
        section.querySelector(".remove-section").disabled = sections.length < 2;
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
    document.querySelectorAll(".diagnostics-section").forEach(wireSection);
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
