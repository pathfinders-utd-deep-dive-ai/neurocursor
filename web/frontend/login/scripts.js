if (localStorage.getItem("username")) {
    window.location.href = "/home/";
}

document.getElementById('login').addEventListener('submit', function(event) {
    event.preventDefault();
    // TODO: Reimplement data, but my way (prob just a massive line in JSON.stringify())
    fetch('/api/login/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dataObject)
    })
    .then(response => response.text())
    .then(result => {
        if (result == "True") {
            localStorage.setItem("username", dataObject.username);
            window.location.href = "/home/";
        } else {
            alert("Login failed");
        }
    })
});