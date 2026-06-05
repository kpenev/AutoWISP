const configFileInput = document.getElementById("config-file");
const configTextArea = document.getElementById("custom-config");
const messageDisplay = document.getElementById("config-file-message");

configFileInput.addEventListener("change", updateConfig);

const updateProjectSubmit = document.getElementById("update-project-config")
const projectForm = document.getElementById("create-project")
const textInputs = projectForm.getElementsByTagName("input")

for (const input of textInputs) {
    input.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent default form submission
            // Trigger the desired button's click event or submit the form directly
            updateProjectSubmit.click(); 
            // Or: document.getElementById('myForm').submit(); 
        }
    });
}

function updateConfig() {
    const file = configFileInput.files[0];
    configTextArea.value = "";
    messageDisplay.textContent = "";

    // Validate file existence and type
    if (!file) {
        showMessage("No file selected. Please choose a file.", "error");
        return;
    }

    // Read the file
    const reader = new FileReader();
    reader.onload = () => {
        configTextArea.value = getConfig(reader.result);
    };
    reader.onerror = () => {
        showMessage("Error reading the file. Please try again.", "error");
    };
    reader.readAsText(file);


}

// Displays a message to the user
function showMessage(message, type) {
  messageDisplay.textContent = message;
  messageDisplay.style.color = type === "error" ? "red" : "green";
}

// Return the full contents of the uploaded config file. The server-side
// ``create_project_view`` now handles the full config-file syntax via
// ``parse_config_overwrites``, so no client-side filtering is needed.
function getConfig(fileText) {
    if (fileText.length === 0) {
        showMessage("Selected file is empty.", "error");
        return "";
    }

    showMessage("Configuration loaded successfully.", "success");
    return fileText;
}
