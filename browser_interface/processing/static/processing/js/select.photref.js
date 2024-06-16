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
        image.width = image.naturalWidth;
    } else if ( image.width > parent_width 
                && 
                new_width < parent_width ) {
        image.width = parent_width;
    } else {
        image.width = new_width;
    }
    image.height = Math.round(image.naturalHeight 
                              * 
                              image.width 
                              / 
                              image.naturalWidth);
}

image = document.getElementById("main-image").onwheel = adjustZoom;
