
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

//Extract sources and display the number detected.
async function updateDetectedStars(starfindURL)
{
    const data = await showImageLocations(
        starfindURL,
        getExtractParams(),
        true
    );
    if ( data && Array.isArray(data.stars) && !data.message ) {
        document.getElementById("detected-star-count").textContent =
            data.stars.length;
        document.getElementById("projected-catalog-count").textContent = "N/A";
    }
}

//Project the catalog and display the number of projected sources.
async function updateProjectedCatalog(projectCatalogURL)
{
    const data = await showImageLocations(
        projectCatalogURL,
        getExtractParams(),
        false,
        {"shape": "circle", "r": 8.0, "color": "#f00"}
    );
    if ( data && Array.isArray(data.stars) && !data.message ) {
        document.getElementById("projected-catalog-count").textContent =
            data.stars.length;
    }
}

