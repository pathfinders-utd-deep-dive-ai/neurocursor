<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $raw_input = file_get_contents('php://input');
    
    $data = json_decode($raw_input, true);
    
    $username = htmlspecialchars($data["username"]);
    $password = password_hash($data["password"], PASSWORD_DEFAULT);
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