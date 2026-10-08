(function() {
    const stepScroll = document.getElementById("processing-step-scroll");

    if (!stepScroll) {
        return;
    }

    stepScroll.scrollLeft = 0;
    stepScroll.addEventListener("wheel", function(event) {
        if (stepScroll.scrollWidth <= stepScroll.clientWidth) {
            return;
        }

        stepScroll.scrollLeft += event.deltaY || event.deltaX;
        event.preventDefault();
    }, {passive: false});
})();
