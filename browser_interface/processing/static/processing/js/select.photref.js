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

var image = document.getElementById("main-image")
image.addEventListener("wheel", adjustZoom);
image.addEventListener("mousedown", panStart);
image.addEventListener("mouseup", panStop);
image.posX = 0;
image.posY = 0;
placeImage();
