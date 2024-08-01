function selectSymbol(event)
{
    let marker = event.currentTarget.className.baseVal.split(" ")[1];
    let master_id = event.currentTarget.parentElement.id.split(":")[1];
    alert("Clicked element ID: " + event.currentTarget.id + ", class: " + event.currentTarget.className.baseVal + ". Selecting button with ID: marker-button:" + master_id);

    let button = document.getElementById("marker-button:" + master_id);
    alert("Button: " + button);
    alert("Button child: " + button.firstChild);
    while (button.firstChild) {
        button.removeChild(button.firstChild);
    }
    button.appendChild(event.currentTarget.cloneNode(true));
}

const plotSymbols = document.getElementsByClassName("plot-marker");
for ( const symbol of plotSymbols ) {
    if ( symbol.parentElement.className == "dropdown-content" )
        symbol.addEventListener("click", selectSymbol);
}
const markerButtons = document.getElemenstByClassName("dropbtn");
for ( const button of markerButtons ) {
}
