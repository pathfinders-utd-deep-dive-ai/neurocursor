if (!localStorage.getItem("username")) {
    window.location.href = "/login/";
}

const username = localStorage.getItem("username");

const canvas = document.getElementById("myCanvas");
if (!canvas) {
    throw new Error('NeuroCursor could not find the canvas element with id "myCanvas".');
}

const ctx = canvas.getContext("2d");
if (!ctx) {
    throw new Error("NeuroCursor could not create a 2D canvas context.");
}

const colorBackground = "#FFFFFF";
const gridColor = "rgba(15, 150, 150, 0.1)";
const dotColor = "#0E9594";
const dotColorTwo = "#D64545";
const LABEL_INK = "#FFFFFF";
const shadowMain = "rgba(15,150,150,0.5)";
const shadowStart = "rgba(215,70,70,0.5)";

const MOVEMENTS_PER_ATTEMPT = 5;
const SAMPLE_INTERVAL_MS = 1000 / 60;

let currentX = 0;
let currentY = 0;
let isClicked = 0;
let buttonsClicked = 0;
let data = [];
let timeOffset = 0;
let loopRunning = false;

let distToStartX = Number.POSITIVE_INFINITY;
let distToStartY = Number.POSITIVE_INFINITY;
let distToButtonX = Number.POSITIVE_INFINITY;
let distToButtonY = Number.POSITIVE_INFINITY;

let activeButton = {};

function updateProgress() {
    fetch("/api/data/get/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ username })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Progress request failed with status ${response.status}.`);
            }
            return response.json();
        })
        .then(result => {
            const progressText = document.getElementById("progress-text");
            if (progressText) {
                progressText.innerText = `${result.length}/20 Sessions Complete`;
            }
        })
        .catch(error => {
            console.error("Unable to update progress:", error);
        });
}

function paintBackdrop() {
    ctx.fillStyle = colorBackground;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;

    const step = 40;

    for (let x = 0.5; x <= canvas.width; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }

    for (let y = 0.5; y <= canvas.height; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
}

function isHover(button) {
    if (
        !button ||
        !Number.isFinite(button.x) ||
        !Number.isFinite(button.y) ||
        !Number.isFinite(button.width) ||
        !Number.isFinite(button.height)
    ) {
        return false;
    }

    const distX = Math.abs(currentX - button.x);
    const distY = Math.abs(currentY - button.y);

    return distX <= button.width / 2 && distY <= button.height / 2;
}

function drawRect(x, y, width, height, color, label) {
    ctx.fillStyle = color;

    const rx = x - width / 2;
    const ry = y - height / 2;

    if (typeof ctx.roundRect === "function") {
        ctx.beginPath();
        ctx.roundRect(rx, ry, width, height, 12);
        ctx.fill();
    } else {
        ctx.fillRect(rx, ry, width, height);
    }

    if (label) {
        ctx.fillStyle = LABEL_INK;
        ctx.font = "600 20px Arial, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, x, y + 1);
    }
}

function renderTheFrame() {
    paintBackdrop();

    const hovering = isHover(activeButton);

    ctx.save();

    if (hovering && Number.isFinite(activeButton.x)) {
        ctx.shadowColor =
            activeButton.color === dotColorTwo ? shadowStart : shadowMain;
        ctx.shadowBlur = 20;
        ctx.shadowOffsetY = 3;
        ctx.shadowOffsetX = 2;
    }

    if (Number.isFinite(activeButton.x)) {
        drawRect(
            activeButton.x,
            activeButton.y,
            activeButton.width,
            activeButton.height,
            activeButton.color,
            activeButton.label
        );
    }

    ctx.restore();
    requestAnimationFrame(renderTheFrame);
}

function updatePointerPosition(event) {
    const rect = canvas.getBoundingClientRect();

    // Convert browser/client coordinates to canvas coordinates. This remains
    // correct when CSS scales the displayed canvas.
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    currentX = (event.clientX - rect.left) * scaleX;
    currentY = (event.clientY - rect.top) * scaleY;
}

window.addEventListener("pointermove", updatePointerPosition);
window.addEventListener("pointerrawupdate", updatePointerPosition);

window.addEventListener("pointerdown", event => {
    updatePointerPosition(event);
    isClicked = 1;
});

window.addEventListener("pointerup", event => {
    updatePointerPosition(event);
    isClicked = 0;
});

window.addEventListener("pointercancel", () => {
    isClicked = 0;
});

window.addEventListener("blur", () => {
    isClicked = 0;
});

function getRandomInt(min, max) {
    const minCeiled = Math.ceil(min);
    const maxFloored = Math.floor(max);

    return Math.floor(
        Math.random() * (maxFloored - minCeiled + 1) + minCeiled
    );
}

function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function placeStartButton() {
    activeButton = {
        x: 700,
        y: 300,
        width: 100,
        height: 50,
        color: dotColorTwo,
        label: "START"
    };
}

function placeMovementTarget() {
    activeButton = {
        x: getRandomInt(200, 1200),
        y: getRandomInt(200, 400),
        width: 50,
        height: 50,
        color: dotColor,
        label: String(buttonsClicked + 1)
    };
}

function recordSample() {
    if (
        !Number.isFinite(activeButton.x) ||
        !Number.isFinite(activeButton.y)
    ) {
        return;
    }

    distToButtonX = currentX - activeButton.x;
    distToButtonY = currentY - activeButton.y;

    const elapsedTime = performance.now() - timeOffset;

    data.push({
        // Primary fields used by the updated model.py parser.
        time: elapsedTime,
        cursor_x: currentX,
        cursor_y: currentY,
        target_x: activeButton.x,
        target_y: activeButton.y,
        relative_x: distToButtonX,
        relative_y: distToButtonY,
        button_state: isClicked,

        // Identifies which of the five sequential target movements this
        // sample belongs to. Values are 0, 1, 2, 3, and 4.
        movement_index: buttonsClicked,

        // Extra context that may be useful for validation or later features.
        target_width: activeButton.width,
        target_height: activeButton.height,
        canvas_width: canvas.width,
        canvas_height: canvas.height,

        // Legacy representation retained so old data-processing code still
        // works while the new model uses the named fields above.
        coords: [distToButtonX, distToButtonY, isClicked]
    });
}

async function saveAttempt() {
    const response = await fetch("/api/data/save/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            username,
            data
        })
    });

    if (!response.ok) {
        const responseText = await response.text();
        throw new Error(
            `Data save failed with status ${response.status}: ${responseText}`
        );
    }

    return response.text();
}

async function startLoop() {
    if (loopRunning) {
        return;
    }

    loopRunning = true;

    try {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        buttonsClicked = 0;
        data = [];
        distToStartX = Number.POSITIVE_INFINITY;
        distToStartY = Number.POSITIVE_INFINITY;

        await wait(500);
        updateProgress();
        placeStartButton();

        // Wait for the START button to be pressed.
        while (!(isClicked === 1 && isHover(activeButton))) {
            distToStartX = currentX - activeButton.x;
            distToStartY = currentY - activeButton.y;
            await wait(SAMPLE_INTERVAL_MS);
        }

        // Require release of the START click so it is not accidentally
        // recorded as the first target click.
        while (isClicked === 1) {
            await wait(SAMPLE_INTERVAL_MS);
        }

        placeMovementTarget();
        timeOffset = performance.now();

        await mainLoop();
    } catch (error) {
        console.error("NeuroCursor collection error:", error);
        alert("The session could not be saved. Please check your connection and try again.");
    } finally {
        loopRunning = false;
    }

    // Begin the next attempt after the current one has fully completed.
    startLoop();
}

async function mainLoop() {
    while (buttonsClicked < MOVEMENTS_PER_ATTEMPT) {
        recordSample();

        if (isClicked === 1 && isHover(activeButton)) {
            // Continue recording while the valid target click is held down.
            while (isClicked === 1) {
                await wait(SAMPLE_INTERVAL_MS);
                recordSample();
            }

            buttonsClicked += 1;

            if (buttonsClicked < MOVEMENTS_PER_ATTEMPT) {
                placeMovementTarget();
            }
        }

        await wait(SAMPLE_INTERVAL_MS);
    }

    const result = await saveAttempt();
    console.log("Attempt saved:", result);
    updateProgress();
}

renderTheFrame();
startLoop();