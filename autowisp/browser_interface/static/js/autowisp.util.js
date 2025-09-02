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


