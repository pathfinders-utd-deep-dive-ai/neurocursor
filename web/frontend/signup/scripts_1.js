document.getElementById('signup').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    const dataObject = Object.fromEntries(formData.entries());
    fetch('/api/signup/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dataObject)
    })
    .then(response => response.text())
    .then(result => {
        console.log(result);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});