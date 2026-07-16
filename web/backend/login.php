<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $raw_input = file_get_contents('php://input');    
    $username = htmlspecialchars(json_decode($raw_input, true)["username"]);
    $password = json_decode($raw_input, true)["password"];
    $users_data = json_decode(file_get_contents("../users.json"), true);

    if (array_key_exists($username, $users_data)) {
        if (password_verify($password, $users_data[$username])) {
            echo "True";
        } else {
            echo "False";
        }
    } else {
        echo "null";
    }
}
?>