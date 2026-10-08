function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}


async function postJson(targetURL, data)
{
    let csrftoken = getCookie('csrftoken');
    let headers = new Headers();
    headers.append('X-CSRFToken', csrftoken);
    headers.append("Content-type", "application/json; charset=UTF-8")
    return await fetch(targetURL, {
        method: "POST",
        body: JSON.stringify(data),
        headers: headers,
        credentials: 'include'
    });
}


function showSVG(data, parentId)
{
    let parentElement = document.getElementById(parentId);
    for ( child of parentElement.children )
        if ( child.tagName.toUpperCase() == "SVG" )
            parentElement.removeChild(child);

    parentElement.innerHTML = data["plot_data"] + parentElement.innerHTML;
    delete data["plot_data"];
    return data
}

function stripUnits(quantity)
{
    while ( isNaN(Number(quantity)) ) 
        quantity = quantity.slice(0, -1);
    return Number(quantity);
}

function setFigureSize(parentId)
{
    let fullRect = document
        .getElementById("active-area")
        .getBoundingClientRect();
    let figureParent = document.getElementById(parentId);
    let figure = figureParent.children[0];
    let width = figure.getAttribute("width");
    let aspectRatio = (stripUnits(figure.getAttribute("width"))
                       / 
                       stripUnits(figure.getAttribute("height")));
    let parentBoundingRect = figureParent.getBoundingClientRect();
    maxHeight = (fullRect.top
                 +
                 fullRect.height
                 -
                 parentBoundingRect.top)
    figureParent.style.height = maxHeight + "px";
    figureParent.style.minHeight = maxHeight + "px";
    figureParent.style.maxHeight = maxHeight + "px";
    figureParent.style.padding = "0px";
    figureParent.style.margin = "0px";
    figure.setAttribute("height", 
                        Math.min(maxHeight, 
                                 parentBoundingRect.width 
                                 / 
                                 aspectRatio ));
    figure.setAttribute("width",
                        Math.min(parentBoundingRect.width,
                                 maxHeight * aspectRatio));
}

function updateFigure()
{
    console.log("Updating figure");
    let param;
    if (typeof updateFigure.getParam === 'function') {
        console.log("Getting parameters");
        param = updateFigure.getParam();
    }

    postJson(updateFigure.url, param)
        .then((response) => {
            console.log(response);
            return response.json();
        })
        .then((data) => {
            console.log(data);
            updateFigure.callback(data);
        })
        .catch(function(error) {
            alert("Updating plot failed: " + error);
        });

}


function handleDropdownClick(event)
{
    // A marker menu opens on a click and stays open until something
    // closes it, like every other control on a series row. Opening on
    // hover meant brushing past a button opened a menu nobody asked for,
    // and choosing from one meant keeping the pointer inside a narrow
    // strip the whole way down.
    //
    // One listener on the document rather than one per menu, so a menu
    // rendered after the page was built needs no wiring. In the capture
    // phase, so that it is reached even where a click is stopped from
    // bubbling -- which it is on a series row, where a click on a control
    // must not also toggle whether the row is drawn.
    //
    // Closing first and then deciding whether to open is what makes a
    // second click on the same button close its menu, and a click
    // anywhere else close it without opening another.
    const open = document.querySelector(".dropdown.open");
    if ( open )
        open.classList.remove("open");

    const button = event.target.closest(".dropbtn");
    if ( button && button.closest(".dropdown") !== open )
        button.closest(".dropdown").classList.add("open");
}

document.addEventListener("click", handleDropdownClick, true);

