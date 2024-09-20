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
}
