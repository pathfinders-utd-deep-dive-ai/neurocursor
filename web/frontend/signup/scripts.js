document.getElementById('generate').addEventListener('submit', function(event) {
    event.preventDefault();
    document.querySelector('input[name="username"]').value = `cursor_${Math.random().toString(36).slice(2, 8)}`;
});

document.getElementById('signup').addEventListener('submit', function(event) {
    event.preventDefault();
    fetch('/api/signup/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(Object.fromEntries(new FormData(this).entries()))
    })
    .then(response => response.text())
    .then(result => {
        if (result == "True") {
            alert("Signup succeeded! Please proceed to login.")
        } else {
            console.log(result)
            alert("Signup failed")
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});