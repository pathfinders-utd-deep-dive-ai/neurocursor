<?php
// Copied from Stackify display PHP errors
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $raw_input = file_get_contents("php://input");
    $username = htmlspecialchars(json_decode($raw_input, true)["username"]);
    $data = json_decode($raw_input, true)["data"];
    $the_mouse_data = json_decode(file_get_contents("../data.json"), true);

    $the_mouse_data[$username] = $data;
    file_put_contents("../data.json", json_encode($the_mouse_data));
}
?>