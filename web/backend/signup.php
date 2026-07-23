<?php
// Copied from Stackify display PHP errors
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $raw_input = file_get_contents('php://input');    
    $username = hash_hmac("sha256", json_decode($raw_input, true)["username"], file_get_contents("../.env"));
    $users_data = json_decode(file_get_contents("../users.json"), true);

    if (!array_key_exists($username, $users_data)) {
        $users_data[$username] = "";
        echo "True";
    } else {
        echo "False";
    }

    file_put_contents("../users.json", json_encode($users_data));
}
?>