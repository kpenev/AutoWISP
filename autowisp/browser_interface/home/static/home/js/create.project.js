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

//Return the part of the configuration to change given a config file
function getConfig(fileText) {
    const lines = fileText.split("\n");
    let configSection = [];

    for (let line of lines) {
        split = line.split('=')
        if ( split.length == 2 ) {
            key = split[0].trim();
            value = split[1].trim();
            if ( !key.endsWith('-fname') 
                 && !key.endsWith('-fname-format') 
                 && !key.endsWith('-catalog') 
                 && key != 'split-channels'
               ) {
                configSection.push(split[0].trim() + " = " + split[1].trim());
            }
        }
    }

    if (configSection.length === 0) {
        showMessage("No configuration section found in the file.", "error");
        return "";
    }

    showMessage("Configuration loaded successfully.", "success");
    return configSection.join("\n");
}
