//Change the displayed histogrames up by one.
function histScrollUp(event)
{
    if ( histParent.firstVisible == histParent.children.length - 1 ) {
        document.getElementById(
            "hist-scroll-down"
        ).addEventListener(
            "click",
            histScrollDown
        );
    }
    histParent.firstVisible = histParent.firstVisible - 1;
    histParent.shift = (
        histParent.shift 
        + 
        histParent.children[histParent.firstVisible].getBoundingClientRect().height
    );
    histParent.style.top = histParent.shift + "px";
    if ( histParent.firstVisible == 0 ) {
        document.getElementById(
            "hist-scroll-up"
        ).removeEventListener(
            "click",
            histScrollUp
        );
    }
}

//Change the displayed histogrames down by one.
function histScrollDown(event)
{
    if ( histParent.firstVisible == 0 ) {
        document.getElementById(
            "hist-scroll-up"
        ).addEventListener(
            "click",
            histScrollUp
        );
    }
    histParent.shift = (
        histParent.shift
        - 
        histParent.children[histParent.firstVisible].getBoundingClientRect().height
    );
    histParent.style.top = histParent.shift + "px";
    histParent.firstVisible = histParent.firstVisible + 1;
    if ( histParent.firstVisible == histParent.children.length - 1 ) {
        document.getElementById(
            "hist-scroll-down"
        ).removeEventListener(
            "click",
            histScrollDown
        );
    }
}

//Prepare to respond to user dragging a histogram.
function histDragStart(event)
{
    event.preventDefault();
    histDragEnd.target = event.target;
    while ( histDragEnd.target.parentElement != histParent ) {
        histDragEnd.target = histDragEnd.target.parentElement;
    }
    histParent.addEventListener("mouseup", histDragEnd);
}

//Update the histogram order after a user has dragged and dropped one.
function histDragEnd(event)
{
    event.preventDefault();
    let i = 0;
    while ( histParent.children[i].getBoundingClientRect().top 
            < 
            event.clientY 
            &&
            i < histParent.children.length) {
        i+= 1;
    }
    histParent.insertBefore(histDragEnd.target, histParent.children[i]);

    histParent.removeEventListener("mouseup", histDragEnd);
}

//Prepare to respond to user dragging histogram/image separator.
function resizeHistStart(event)
{
    event.preventDefault();
    document.getElementById("full-view").addEventListener("mousemove",
                                                          resizeHist);
    document.getElementById("full-view").addEventListener("mouseup",
                                                          resizeHistEnd);
}

//Adjust the size of the histograms in response to user dragging separator
function resizeHist(event)
{
    event.preventDefault();
    let fullRect = document.getElementById("full-view").getBoundingClientRect();
    let sideBar = document.getElementById("side-bar")
    sideBar.style.width = Math.max(
        document.getElementById("vert-hist-sep").getBoundingClientRect().width,
        (fullRect.right 
         - 
         Math.max(fullRect.left + 100, event.clientX))
    ) + "px";
    sideBar.style.minWidth = sideBar.style.width
}

//The user has released the image/histogram separator.
function resizeHistEnd(event)
{
    event.preventDefault();
    document.getElementById("full-view").removeEventListener("mousemove",
                                                             resizeHist);
    document.getElementById("full-view").removeEventListener("mouseup",
                                                             resizeHistEnd);
}

//Prepare to respond to user interacting with the photref selection view.
function initView(viewConfig)
{
    if ( viewConfig === undefined ) {
        initFITS();
        viewConfig = {
            "image": getFITSConfig(),
            "histograms": {
                "firstVisible": 0,
                "width": document.getElementById("side-bar").style.width,
                "order": [...Array(histParent.children.length).keys()]
            }
        };
    } else {
        initFITS(viewConfig["image"]);
    }

    for( let i = 0; i < histParent.children.length; i++) {
        histParent.children[i].addEventListener("mousedown", histDragStart);
        histParent.children[i].origPosition = i;
    }
    let histOrder = viewConfig.histograms.order;
    let originalOrder = Array.from(histParent.children);
    for( let i = 0; i < histParent.children.length; ++i ) {
        if ( i != histOrder[i] ) {
            histParent.insertBefore(originalOrder[histOrder[i]],
                                    histParent.children[i]);
        }
    }
    histParent.firstVisible = viewConfig.histograms.firstVisible;

    histParent.shift = 0;
    for( let i = 0; i < histParent.firstVisible; ++i ) {
        histParent.shift = (
            histParent.shift 
            + 
            histParent.children[i].getBoundingClientRect().height
        );
    }

    let sideBar = document.getElementById("side-bar")
    sideBar.style.width = viewConfig.histograms.width;
    sideBar.style.minWidth = sideBar.style.width

    document.getElementById(
        "hist-scroll-down"
    ).addEventListener(
        "click",
        histScrollDown
    );
    document.getElementById("resize-hist").addEventListener("mousedown",
                                                            resizeHistStart);
}

//Submit the currently configured view when changing scale, range or image
async function updateView(change)
{
    viewConfig = {
        "change": change,
        "image": getFITSConfig(),
        "histograms": {
            "firstVisible": histParent.firstVisible,
            "width": document.getElementById("side-bar").style.width,
            "order": []
        }
    };

    for( let i = 0; i < histParent.children.length; i++) {
        viewConfig.histograms.order.push(histParent.children[i].origPosition);
    }
    postJson(updateView.URL, viewConfig).then(
        function() {
            location.reload();
        }
    );
}

var histParent = document.getElementById("hist-parent");
