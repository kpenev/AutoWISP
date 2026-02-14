function toggleAll(source) {
    var checkboxes = document.querySelectorAll(
        'input[name="project_ids"]'
    );
    for (var i = 0; i < checkboxes.length; i++) {
        checkboxes[i].checked = source.checked;
    }
}

function deleteSelected() {
    var checked = document.querySelectorAll(
        'input[name="project_ids"]:checked'
    );
    if (checked.length === 0) {
        alert("No projects selected.");
        return;
    }
    var count = checked.length;
    if (!confirm("Delete " + count + " selected project(s)?")) {
        return;
    }
    document.getElementById("project-form").submit();
}
