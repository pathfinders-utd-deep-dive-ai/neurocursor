if (localStorage.getItem("username")) {
    window.location.href = "/home/";
}

document.getElementById('login').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    const dataObject = Object.fromEntries(formData.entries());
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
    .catch(error => {
        console.error('Error:', error);
    });
});