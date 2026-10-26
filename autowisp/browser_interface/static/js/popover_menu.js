// Opens a .popover-menu (see popover_menu.css) on hover, and lets a page
// wait until none is open, e.g. to reload without closing a menu under the
// user, and losing what they typed in it.

// Call back once no popover menu is open: now, if none is.
function whenPopoverMenusClosed(callback) {
    var open = document.querySelector('.popover-menu:popover-open');
    if (!open) {
        callback();
        return;
    }
    open.addEventListener('toggle', function onToggle(event) {
        if (event.newState !== 'closed') return;
        open.removeEventListener('toggle', onToggle);
        // Another may have opened as this one closed.
        whenPopoverMenusClosed(callback);
    });
}

// A menu opens while the pointer is over its button or the menu itself,
// and closes shortly after it leaves both: the grace covers a diagonal move
// brushing past the edge. A click on the button keeps it open until clicked
// again, Escape or a click outside; so does a field in it having the focus,
// e.g. a name half typed.
(function () {
    var closeDelay = 150;

    function setUp(button) {
        var menu = document.getElementById(
            button.getAttribute('popovertarget')
        );
        if (!menu || !menu.classList.contains('popover-menu')) return;
        var pinned = false;
        var closeTimer = null;

        function isOpen() {
            return menu.matches(':popover-open');
        }
        function open() {
            clearTimeout(closeTimer);
            if (!isOpen()) menu.showPopover();
        }
        function closeSoon() {
            clearTimeout(closeTimer);
            closeTimer = setTimeout(function () {
                if (isOpen() && !pinned && !menu.contains(document.activeElement)) {
                    menu.hidePopover();
                }
            }, closeDelay);
        }

        button.addEventListener('mouseenter', open);
        button.addEventListener('mouseleave', closeSoon);
        menu.addEventListener('mouseenter', open);
        menu.addEventListener('mouseleave', closeSoon);
        // The button's own toggling would close a menu just opened by
        // hovering over it: a click pins it instead, and unpins it.
        button.addEventListener('click', function (event) {
            event.preventDefault();
            if (isOpen() && pinned) {
                menu.hidePopover();
            } else {
                open();
                pinned = true;
            }
        });
        menu.addEventListener('toggle', function (event) {
            if (event.newState === 'closed') pinned = false;
        });
    }

    document.querySelectorAll('[popovertarget]').forEach(setUp);
})();
