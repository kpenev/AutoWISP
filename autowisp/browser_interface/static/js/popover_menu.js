// The browser opens and closes a .popover-menu by itself (see
// popover_menu.css); this only lets a page wait until none is open, e.g.
// to reload without closing a menu under the user, and losing what they
// typed in it.

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
