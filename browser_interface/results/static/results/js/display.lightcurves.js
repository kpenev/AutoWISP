function updateFigure()
{
    postJson(updateFigure.url)
        .then((response) => {
            console.log(response);
            return response.json();
        })
        .then((data) => {
            console.log(data);
            showSVG(data, "figure-parent");
        })
        .catch(function(error) {
            alert("Updating plot failed: " + error);
        });

}

function initLightcurveDisplay(updateURL)
{
    updateFigure.url = updateURL;
    updateFigure();
}
