<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $raw_input = file_get_contents('php://input');    
    $username = htmlspecialchars(json_decode($raw_input, true)["username"]);
    $password = password_hash(json_decode($raw_input, true)["password"], PASSWORD_DEFAULT);
    $users_data = json_decode(file_get_contents("../users.json"), true);

    if (!array_key_exists($username, $users_data)) {
        $users_data[$username] = $password;
        echo "True";
    } else {
        echo "False";
    }

    file_put_contents("../users.json", json_encode($users_data));
}
?>