//Display the given sources as markers on top of the FITS image.
function markExtractedSources(sources, replace, marker)
{
    if ( marker === undefined ) {
        marker = {
            "shape": "circle",
            "r": 5.0
        }
    }
    console.log(sources);
    const regions = [];
    for ( let i = 0; i < sources.length; i++ ) {
        new_reg = {
            "x": sources[i].x,
            "y": sources[i].y
        };
        for ( let property in marker )
            new_reg[property] = marker[property];
        regions.push(new_reg);
    }
    addRegions(regions, "px", replace);
}

function getExtractParams()
{
    let extractParams = { 
        "srcfind-tool": null,
        "brightness-threshold": null, 
        "filter-sources": null, 
        "max-sources": null
    };

    for ( param in extractParams ) {
        extractParams[param] = document.getElementById(param).value
    }
    return extractParams;
}

//Ask the server for a new list of extracted sources and display them.
async function showSources(starFindURL, replace, marker)
{
    let extractParams = getExtractParams();
    let csrftoken = getCookie('csrftoken');
    let headers = new Headers();
    headers.append('X-CSRFToken', csrftoken);
    headers.append("Content-type", "application/json; charset=UTF-8")
    const response = await fetch(starFindURL, {
        method: "POST",
        body: JSON.stringify(extractParams),
        headers: headers,
        credentials: 'include'
    })
        .then((response) => {
            console.log(response);
            return response.json();
        })
        .then((data) => {
            console.log(data);
            if ( data.stars.length == 0 )
                alert(data.message);
            markExtractedSources(data.stars, replace, marker);
        })
        .catch(function(error) {
            alert("Adding sources failed:" + error);
        })
}
