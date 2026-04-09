function navigateToDiagPair()
{
    let selector = document.getElementById("diag-vs-diag-selector");
    let urlTemplate = selector.dataset.urlTemplate;
    let x = document.getElementById("x-diagnostic-selector").value;
    let y = document.getElementById("y-diagnostic-selector").value;
    window.location.href = urlTemplate
        .replace("XPLACEHOLDER", x)
        .replace("YPLACEHOLDER", y);
}
