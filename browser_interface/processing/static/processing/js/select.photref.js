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

function pan(event)
{
    event.preventDefault();
    let shiftX = event.clientX - pan.startX;
    let shiftY = event.clientY - pan.startY;

    image.posX = pan.imageStartX + shiftX;
    image.posY = pan.imageStartY + shiftY;
    placeImage();
}

function panStart(event)
{
    event.preventDefault();
    pan.startX = event.clientX;
    pan.startY = event.clientY;
    pan.imageStartX = image.posX;
    pan.imageStartY = image.posY;
    image.addEventListener("mousemove", pan);
}

function panStop(event)
{
    event.preventDefault();
    image.removeEventListener("mousemove", pan); 
}

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

function histDragStart(event)
{
    event.preventDefault();
    histDragEnd.target = event.target;
    while ( histDragEnd.target.parentElement != histParent ) {
        histDragEnd.target = histDragEnd.target.parentElement;
    }
    histParent.addEventListener("mouseup", histDragEnd);
}

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
    alert("Inserting " + histDragEnd.target + " before histogram " + i);
    histParent.insertBefore(histDragEnd.target, histParent.children[i]);

    histParent.removeEventListener("mouseup", histDragEnd);
}

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
    }
}
var image = document.getElementById("main-image")
var histParent = document.getElementById("hist-parent");
init();
