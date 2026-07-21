if (localStorage.getItem("username")) {
    window.location.href = "/home/";
}

document.getElementById('login').addEventListener('submit', function(event) {
    event.preventDefault();
    fetch('/api/login/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(Object.fromEntries(new FormData(this).entries()))
    })
    .then(response => response.text())
    .then(result => {
        if (result == "True") {
            localStorage.setItem("username", Object.fromEntries(new FormData(this).entries()).username);
            window.location.href = "/home/";
        } else {
            console.log(result)
            alert("Login failed");
        }
    })
});