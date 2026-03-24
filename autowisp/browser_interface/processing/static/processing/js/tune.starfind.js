// Read a numeric input field and validate it is a finite number.
function _parseNumber(fieldId, fieldLabel) {
    const fieldValue = document.getElementById(fieldId).value;
    const parsedValue = Number(fieldValue);
    if (!Number.isFinite(parsedValue)) {
        alert(fieldLabel + " must be a valid number.");
        return null;
    }
    return parsedValue;
}

// Show/hide threshold controls based on the selected threshold mode.
function setThresholdModeUI(mode) {
    const isQuantileMode = mode === "quantile";
    document.getElementById("brightness-threshold-row").style.display =
        isQuantileMode ? "none" : "";
    document.getElementById("brightness-threshold-break").style.display =
        isQuantileMode ? "none" : "";
    document.getElementById("brightness-quantile-row").style.display =
        isQuantileMode ? "" : "none";
    document.getElementById("brightness-quantile-break").style.display =
        isQuantileMode ? "" : "none";
    document.getElementById("brightness-quantile-scale-row").style.display =
        isQuantileMode ? "" : "none";
    document.getElementById("brightness-quantile-scale-break").style.display =
        isQuantileMode ? "" : "none";
}

// Collect and validate source-extraction parameters from the form.
function getExtractParams() {
    const thresholdMode = document.getElementById("threshold-mode").value;
    let extractParams = {
        "srcfind-tool": document.getElementById("srcfind-tool").value,
        "threshold-mode": thresholdMode,
        "filter-sources": document.getElementById("filter-sources").value,
        "max-sources": document.getElementById("max-sources").value,
    };

    // Manual absolute threshold mode.
    if (thresholdMode === "brightness-threshold") {
        const threshold = _parseNumber(
            "brightness-threshold",
            "Brightness threshold"
        );
        if (threshold === null) {
            return null;
        }
        if (threshold <= 0) {
            alert("Brightness threshold must be positive.");
            return null;
        }
        extractParams["brightness-threshold"] = threshold;
    } else {
        // Quantile-based threshold mode.
        const quantile = _parseNumber("brightness-quantile", "Quantile");
        if (quantile === null) {
            return null;
        }
        if (quantile < 0 || quantile > 1) {
            alert("Quantile must be between 0 and 1.");
            return null;
        }

        const quantileScale = _parseNumber(
            "brightness-quantile-scale",
            "Quantile scale"
        );
        if (quantileScale === null) {
            return null;
        }
        if (quantileScale <= 0) {
            alert("Quantile scale must be positive.");
            return null;
        }

        extractParams["brightness-quantile"] = quantile;
        extractParams["brightness-quantile-scale"] = quantileScale;
    }

    return extractParams;
}

// Refresh extracted source overlays using current form parameters.
function updateExtractedSources(starfindURL) {
    const params = getExtractParams();
    if (!params) {
        return;
    }
    showImageLocations(starfindURL, params, true);
}

// Refresh projected catalog overlays using current form parameters.
function updateProjectedCatalog(projectCatURL) {
    const params = getExtractParams();
    if (!params) {
        return;
    }
    showImageLocations(
        projectCatURL,
        params,
        false,
        { "shape": "circle", "r": 8.0, "color": "#f00" }
    );
}

// Initialize the threshold controls when the page loads.
setThresholdModeUI(document.getElementById("threshold-mode").value);

