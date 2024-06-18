//Place the main image relative to its parent per its posX and posY attributes
function placeImage()
{
    let boundingRect = document.getElementsByClassName(
        "main-parent"
    )[0].getBoundingClientRect();

    image.style.left = (
        image.posX 
        + 
        Math.round((boundingRect.width - image.width) / 2)
    ) + "px";
    image.style.top = (
        image.posY 
        + 
        Math.round((boundingRect.height - image.height) / 2)
    ) + "px";
}

//Change the zoom level of the main image.
function adjustZoom(event)
{
    event.preventDefault();
    let image = document.getElementById("main-image");
    let parentWidth = document.getElementsByClassName("main-parent")[0].getBoundingClientRect().width;
    let step = Math.round(image.width / 100);
    newWidth = Math.max(
        100,
        image.width + event.deltaY * step
    );
    newWidth = Math.min(newWidth, 100 * image.naturalWidth)
    if ( image.width > image.naturalWidth && newWidth < image.naturalWidth ) {
        newWidth = image.naturalWidth;
    } else if ( image.width > parentWidth 
                && 
                newWidth < parentWidth ) {
        newWidth = parentWidth;
    } 

    let scale = newWidth / image.width
    image.posX = image.posX * scale;
    image.posY = image.posY * scale;
    image.width = newWidth;
    image.height = Math.round(image.naturalHeight 
                              * 
                              image.width 
                              / 
                              image.naturalWidth);
    placeImage();
}

//Change the displayed portion of the main image in response to dragging.
function pan(event)
{
    event.preventDefault();
    let shiftX = event.clientX - pan.startX;
    let shiftY = event.clientY - pan.startY;

    image.posX = pan.imageStartX + shiftX;
    image.posY = pan.imageStartY + shiftY;
    placeImage();
}

//Prepare to respond to the user dragging the main image. 
function panStart(event)
{
    event.preventDefault();
    pan.startX = event.clientX;
    pan.startY = event.clientY;
    pan.imageStartX = image.posX;
    pan.imageStartY = image.posY;
    image.addEventListener("mousemove", pan);
}

//The user has released the main image after dragging it.
function panStop(event)
{
    event.preventDefault();
    image.removeEventListener("mousemove", pan); 
}

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
        viewConfig = {
            "image": {
                "posX": 0, 
                "posY": 0, 
                "width": image.width,
                "height": image.height
            },
            "histograms": {
                "firstVisible": 0,
                "width": document.getElementById("side-bar").style.width,
                "order": [...Array(histParent.children.length).keys()]
            }
        };

    }
    image.addEventListener("wheel", adjustZoom);
    image.addEventListener("mousedown", panStart);
    image.addEventListener("mouseup", panStop);
    image.posX = viewConfig.image.posX;
    image.posY = viewConfig.image.posY;
    image.width = viewConfig.image.width;
    image.height = Math.round(image.naturalHeight 
                              * 
                              image.width 
                              / 
                              image.naturalWidth);
    placeImage();

    let histOrder = viewConfig.histograms.order;
    for( let i = 0; i < histParent.children.length; ++i ) {
        histParent.insertBefore(histParent.children[histOrder[i]],
                                histParent.children[i]);
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
    for( let i = 0; i < histParent.children.length; i++) {
        histParent.children[i].addEventListener("mousedown", histDragStart);
        histParent.children[i].origPosition = i;
    }
    document.getElementById("resize-hist").addEventListener("mousedown",
                                                            resizeHistStart);
}

//Submit the currently configured view when changing scale, range or image
async function updateView(updateURL)
{
    viewConfig = {
        "image": {
            "posX": image.posX, 
            "posY": image.posY, 
            "width": image.width,
            "height": image.height
        },
        "histograms": {
            "firstVisible": histParent.firstVisible,
            "width": document.getElementById("side-bar").style.width,
            "order": []
        }
    };

    for( let i = 0; i < histParent.children.length; i++) {
        viewConfig.histograms.order.push(histParent.children[i].origPosition);
    }
    postJson(updateURL, viewConfig).then(
        function() {
            document.location = updateURL;
        }
    );
}

var image = document.getElementById("main-image")
var histParent = document.getElementById("hist-parent");
