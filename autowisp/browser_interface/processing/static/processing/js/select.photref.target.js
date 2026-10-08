//Rank the candidates by the merit expression just chosen: the choice is
//remembered by the server, and used on every candidate page that follows.
document.getElementById("merit").addEventListener(
    "change",
    function(event) {
        event.target.form.submit();
    }
);
