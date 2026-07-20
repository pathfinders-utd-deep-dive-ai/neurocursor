<?php
// Copied from Stackify display PHP errors
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $raw_input = file_get_contents('php://input');    
    $username = hash_hmac("sha256", json_decode($raw_input, true)["username"], file_get_contents("../../.env"));
    $the_mouse_data = json_decode(file_get_contents("../../data.json"), true);

    if (isset($the_mouse_data[$username])) {
        echo json_encode($the_mouse_data[$username]);
    } else {
        echo json_encode([]);
    }
}