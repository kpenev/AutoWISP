//Display the given sources as markers on top of the FITS image.
function markExtractedSources(sources, marker)
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
    addRegions(regions, "px", true);
}

//Ask the server for a new list of extracted sources and display them.
async function showExtractedSources(starFindURL)
{
    let extract_params = { 
        "srcfind-tool": null,
        "brightness-threshold": null, 
        "filter-sources": null, 
        "max-sources": null
    };

    for ( param in extract_params ) {
        extract_params[param] = document.getElementById(param).value
    }

    let csrftoken = getCookie('csrftoken');
    let headers = new Headers();
    headers.append('X-CSRFToken', csrftoken);
    headers.append("Content-type", "application/json; charset=UTF-8")
    const response = await fetch(starFindURL, {
        method: "POST",
        body: JSON.stringify(extract_params),
        headers: headers,
        credentials: 'include'
    })
        .then((response) => {
            return response.json();
        })
        .then((data) => {
            console.log(data);
            markExtractedSources(data.stars);
        })
        .catch(function(error) {
            alert("Source extraction failed:" + error);
        })
}
