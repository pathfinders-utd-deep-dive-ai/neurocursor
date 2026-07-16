document.getElementById('signup').addEventListener('submit', function(event) {
    event.preventDefault();
    // TODO: Reimplement data, but my way (prob just a massive line in JSON.stringify())
    fetch('/api/signup/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dataObject)
    })
    .then(response => response.text())
    .then(result => {
        if (result == "True") {
            alert("Signup succeeded! Please proceed to login.")
        } else {
            alert("Signup failed")
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});