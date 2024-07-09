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

//Prepare to respond to user interacting with the FITS image.
function initFITS(config)
{
    if ( config === undefined ) {
        config = {
            "posX": 0, 
            "posY": 0, 
            "width": image.width,
            "height": image.height
        };
    }
    image.addEventListener("wheel", adjustZoom);
    image.addEventListener("mousedown", panStart);
    image.addEventListener("mouseup", panStop);
    image.posX = config.posX;
    image.posY = config.posY;
    image.width = config.width;
    image.height = Math.round(image.naturalHeight 
                              * 
                              image.width 
                              / 
                              image.naturalWidth);
    placeImage();
}

function getFITSConfig()
{
    return {
        "posX": image.posX, 
        "posY": image.posY, 
        "width": image.width,
        "height": image.height
    }
}

//Prepare to respond to user interacting with the displayed image.
function initView(viewConfig)
{
    if ( viewConfig === undefined ) {
        initFITS();
        viewConfig = {
            "image": getFITSConfig()
        };
    } else {
        initFITS(viewConfig["image"]);
    }
}

//Submit the currently configured view when changing scale, range or image
async function updateView(change)
{
    viewConfig = {
        "change": change,
        "image": getFITSConfig(),
    };

    postJson(updateView.URL, viewConfig).then(
        function() {
            location.reload();
        }
    );
}

var image = document.getElementById("main-image")
