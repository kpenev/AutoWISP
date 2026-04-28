(function () {
    var key = 'lcars_scroll';
    var scroller = document.getElementById("container");

    window.addEventListener('load', function () {
        var saved = sessionStorage.getItem(key);
        if (saved !== null) {
            sessionStorage.removeItem(key);
            scroller.scrollTop = parseInt(saved, 10);
        }
    });

    document.addEventListener('click', function (e) {
        var link = e.target.closest('a[href]');
        if (link) {
            sessionStorage.setItem(key, scroller.scrollTop);
        }
    });

    document.addEventListener('submit', function () {
        sessionStorage.setItem(key, scroller.scrollTop);
    });
}());
