
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

