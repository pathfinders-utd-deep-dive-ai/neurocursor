<?php
// Copied from Stackify display PHP errors
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $raw_input = file_get_contents("php://input");
    $username = hash_hmac("sha256", json_decode($raw_input, true)["username"], file_get_contents("../.env"));
    $data = json_decode($raw_input, true)["data"];
    $the_mouse_data = json_decode(file_get_contents("../data.json"), true);

    if ($the_mouse_data[$username] != null) {
        $the_mouse_data[$username][] = $data;
    } else {
        $the_mouse_data[$username] = [$data];
    }
    file_put_contents("../data.json", json_encode($the_mouse_data));
}
?>

Normalize performance.now() to 0 before pressing start, add the x/20 in the html and restart session after finish.