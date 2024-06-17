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
    let parent_width = document.getElementsByClassName("main-parent")[0].getBoundingClientRect().width;
    let step = Math.round(image.width / 100);
    new_width = Math.max(
        100,
        image.width + event.deltaY * step
    );
    new_width = Math.min(new_width, 100 * image.naturalWidth)
    if ( image.width > image.naturalWidth && new_width < image.naturalWidth ) {
        new_width = image.naturalWidth;
    } else if ( image.width > parent_width 
                && 
                new_width < parent_width ) {
        new_width = parent_width;
    } 

    let scale = new_width / image.width
    image.posX = image.posX * scale;
    image.posY = image.posY * scale;
    image.width = new_width;
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
    let full_rect = document.getElementById("full-view").getBoundingClientRect();
    let side_bar = document.getElementById("side-bar")
    side_bar.style.width = Math.max(
        document.getElementById("vert-hist-sep").getBoundingClientRect().width,
        (full_rect.right 
         - 
         Math.max(full_rect.left + 100, event.clientX))
    ) + "px";
    side_bar.style.minWidth = side_bar.style.width
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
function init()
{
    image.addEventListener("wheel", adjustZoom);
    image.addEventListener("mousedown", panStart);
    image.addEventListener("mouseup", panStop);
    image.posX = 0;
    image.posY = 0;
    placeImage();

    histParent.firstVisible = 0;
    histParent.shift = 0;
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
async function update(updateURL)
{
    view_config = {
        "image": {
            "posX": image.posX, 
            "posY": image.posY, 
            "width": image.width,
            "height": image.height
        },
        "histograms": {
            "firstVisible": histParent.firstVisible,
            "width": document.getElementById("side-bar").style.width,
        }
    };

    for( let i = 0; i < histParent.children.length; i++) {
        view_config.histograms.order.push(histParent.children[i].origPosition);
    }
    postJson(updateURL, view_config);
}

var image = document.getElementById("main-image")
var histParent = document.getElementById("hist-parent");
init();
