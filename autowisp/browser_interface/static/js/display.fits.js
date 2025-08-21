var svgNS = "http://www.w3.org/2000/svg";

//Place the main image relative to its parent per its posX and posY attributes
function placeImage()
{
    let boundingRect = document.getElementById(
        "img-parent"
    ).getBoundingClientRect();

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

    let regionsElement = document.getElementById("regions");
    if ( regionsElement != null ) {
        regionsElement.style.left = (
            boundingRect.left 
            +
            image.posX
            +
            Math.round((boundingRect.width - image.width) / 2)
        ) + "px";
        regionsElement.style.top = (
            boundingRect.top 
            + 
            image.posY
            +
            Math.round((boundingRect.height - image.height) / 2)
        ) + "px";
    }
}

//Change the zoom level of the main image.
function adjustZoom(event)
{
    event.preventDefault();
    let image = document.getElementById("main-image");
    let parentWidth = document.getElementById("img-parent").getBoundingClientRect().width;
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

    let regionsElement = document.getElementById("regions");
    if ( regionsElement != null ) {
        regionsElement.setAttribute("width", image.width);
        regionsElement.setAttribute("height", image.height);
    }
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

//Transform given regions from pixel to fractional coordinates.
function scaleRegions(regions)
{
    let x_scale = image.naturalWidth / 100.0;
    let y_scale = image.naturalHeight / 100.0;
    let r_scale = Math.max(x_scale, y_scale);
    const result = regions.slice();
    for( let i = 0; i < regions.length; i++) {
        for(let param in regions[i]) {
            if ( param == "x" || param == "width" || param == "rx" ) {
                result[i][param] = regions[i][param] / x_scale;
            } else if ( param == "y" || param == "height" || param == "ry" ) {
                result[i][param] = regions[i][param] / y_scale;
            } else if ( param == "r" ) {
                result[i][param] = regions[i][param] / r_scale;
            }
        }
    }
    return result;
}

function getRegionsElement()
{
    let regionsElement = document.getElementById("regions");
    if ( regionsElement != null ) {
        return regionsElement;
    }
    let mainParent = document.getElementById("img-parent");
    let boundingRect = mainParent.getBoundingClientRect();
    if ( regionsElement == null ) {
        regionsElement = document.createElementNS(svgNS, 'svg');
        regionsElement.id = "regions"
        regionsElement.style.position = "absolute";
        regionsElement.setAttribute("width", image.width);
        regionsElement.setAttribute("height", image.height);
        regionsElement.style.background = '#7777'
        mainParent.appendChild(regionsElement);
        placeImage();
    }
    return regionsElement;
}

//Similar to regions in DS9, but lower left corver of lower left pixel is (0,0)
function addRegions(
                    //The regions to draw. The format is JSON:
                    //{
                    //  "shape": <the name of the shape to mark with>
                    //  "x": <shape center x coordinate>
                    //  "y": <shape center y coordinate>
                    //  ... <other shape parameters e.g. r, width, height> ...
                    //  "color": <color>    
                    //  "linewidth": <linewidth to draw shape with>
                    //}
                    //Color and linewidth are optional and default to "green" 
                    //and 1
                    //
                    //Supported regions are: rect, circle, ellipse, x, +
                    regions, 

                    //Should be either "px" (coordinates are specified in
                    //pixels) or "frac" (coordinates are specified as fraction
                    //of the image width for x and height for y). Default is
                    //"frac"
                    units,
                   
                    //delete previous regions before adding the new ones.
                    replace)
{
    let regionsElement = getRegionsElement();

    if ( units == "px" ) {
        addRegions(scaleRegions(regions));
    }

    if ( replace )
        regionsElement.replaceChildren();
    for ( let reg_ind = 0; reg_ind < regions.length; reg_ind++ ) {
        let reg = regions[reg_ind];
        const elements = [];
        if ( reg.shape == "x" || reg.shape == "+" ) {
            for ( let i = 0; i < 2; i++) {
                elements.push(document.createElementNS(svgNS, "line"));
            }
            let minx = reg.x - reg.width / 2;
            let miny = reg.y - reg.height / 2;
            elements[0].setAttribute("x1", minx + "%");
            elements[0].setAttribute("x2", (minx + reg.width) + "%");
            elements[1].setAttribute("y1", miny + "%");
            elements[1].setAttribute("y2", (miny + reg.height) + "%");

            if ( reg.shape == "x" ) {
                elements[0].setAttribute("y1", miny + "%");
                elements[0].setAttribute("y2", (miny + reg.height) + "%");
                elements[1].setAttribute("x1", (minx + reg.width) + "%");
                elements[1].setAttribute("x2", minx + "%");
            } else {
                elements[0].setAttribute("y1", reg.y + "%");
                elements[0].setAttribute("y2", reg.y + "%");
                elements[1].setAttribute("x1", reg.x + "%");
                elements[1].setAttribute("x2", reg.x + "%");
            }
        } else {
            elements.push(document.createElementNS(svgNS, reg.shape));
            if ( reg.shape == "rect" ) {
                elements[0].setAttribute("x", (reg.x - reg.width / 2) + "%");
                elements[0].setAttribute("y", (reg.y - reg.height / 2) + "%");
                elements[0].setAttribute("width", reg.width + "%");
                elements[0].setAttribute("height", reg.height + "%");
            } else {
                elements[0].setAttribute("cx", reg.x + "%");
                elements[0].setAttribute("cy", reg.y + "%");
                if ( reg.shape == "circle" ) {
                    elements[0].setAttribute("r", reg.r + "%");
                } else if ( reg.shape == "ellipse" ) {
                    elements[0].setAttribute("rx", reg.rx + "%");
                    elements[0].setAttribute("ry", reg.ry + "%");
                }
            }
        }
        for ( let i = 0; i < elements.length; i++ ) {
            if ( reg.linewidth !== undefined ) {
                elements[i].setAttribute("stroke-width", reg.linewidth);
            }
            if ( reg.color === undefined ) {
                elements[i].setAttribute("stroke", "#0f0");
            } else {
                elements[i].setAttribute("stroke", reg.color);
            }
            elements[i].setAttribute("fill-opacity", "0.0");
            regionsElement.appendChild(elements[i]);
        }
    }
}

//Submit the currently c{
//}onfigured view when changing scale, range or image
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
