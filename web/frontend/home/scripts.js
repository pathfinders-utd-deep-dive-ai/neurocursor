var currentX = 0
var currentY = 0
var isClicked = 0
var buttonX = 0
var buttonY = 0
var buttonsClicked = 0
var data = []
let distToStartX = 9999;
let distToStartY = 9999;
let distToButtonX = 9999;
let distToButtonY = 9999;
var activeButton = {};

const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");

const colorBackground = "#FFFFFF";
const gridColor = "rgba(15, 150, 150, 0.1)";
const dotColor = "#0E9594";
const dotColorTwo = "#D64545";
const LABEL_INK = "#FFFFFF";
const shadowMain = "rgba(15,150,150,0.5)"
const shadowStart = "rgba(215,70,70,0.5)"

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
    if(!button) return false;
    let distX = Math.abs(currentX - button.x);
    let distY = Math.abs(currentY - button.y);
    let maxWidthDist = button.width/2;
    let maxHeightDist = button.height/2;
    if(maxWidthDist > distX) {
        if(maxHeightDist > distY) {
            return true;
        }
    }
    return false;
}
function drawRect(x, y, width, height, color, label) {
    ctx.fillStyle = color;
    const rx = x - (width / 2), ry = y - (height / 2);
    if (ctx.roundRect) {
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
    let hovering = isHover(activeButton)
    ctx.save();
    if(hovering && activeButton.x != null) {
        if(activeButton.color == dotColorTwo) {
            ctx.shadowColor = shadowStart;
        }
        else {
            ctx.shadowColor = shadowMain;
        }
        ctx.shadowBlur = 20;
        ctx.shadowOffsetY = 3;
        ctx.shadowOffsetX = 2;
    }
    if(activeButton.x != null) {
        drawRect(activeButton.x, 
                activeButton.y, 
                activeButton.width, 
                activeButton.height, 
                activeButton.color, 
                activeButton.label
            );
    }
        
        ctx.restore();
    //Taken from Google AI Mode Overview- Instructs browser you want to animate something
    requestAnimationFrame(renderTheFrame);
}

window.addEventListener('pointerrawupdate', e => {
  const rect = canvas.getBoundingClientRect();
  
  // 1. Subtract the canvas's left/top offset on the page
  // 2. Multiply by the ratio of internal canvas width vs layout width to fix scaling
  currentX = (e.clientX - rect.left) * (canvas.width / rect.width);
  currentY = (e.clientY - rect.top) * (canvas.height / rect.height);
}); // Taken from Gemini

window.addEventListener('pointerdown', e => { if (e.buttons === 1) isClicked = 1; }); // Taken from Google AI Mode
window.addEventListener('pointerup', () => { isClicked = 0; }); // Taken from Google AI Mode

function refreshPositionBars() {
    if (currentX) {
        let roundedX = Math.round(currentX);
        positionElements.x.textContent = String(roundedX).padStart(4, '0');
    }
    if (currentY) {
        let roundedY = Math.round(currentY);
        positionElements.y.textContent = String(roundedY).padStart(4, '0');
    }
    if (isClicked) {
        positionElements.click.textContent = isClicked == 1 ? 'Down' : 'Idle';
    }
    requestAnimationFrame(refreshPositionBars);
}
refreshPositionBars();


// Taken from Google AI Overview
function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

//Taken from Google AI Overview- Forces program to "wait" to prevent crashing
function sleep(milliseconds) {
    return new Promise(function(resolve) {
        setTimeout(resolve, milliseconds);
    });
}


// Taken from Google AI Overview
function downloadToFile(content, filename, contentType = 'text/plain') {
  // 1. Create a Blob object with the variable's content
  const blob = new Blob([content], { type: contentType });
  
  // 2. Create an invisible anchor element
  const a = document.createElement('a');
  a.style.display = 'none';
  document.body.appendChild(a);
  
  // 3. Create a temporary URL pointing to the Blob object
  const url = window.URL.createObjectURL(blob);
  
  // 4. Set the download destination and filename
  a.href = url;
  a.download = filename;
  
  // 5. Trigger the download automatically
  a.click();
  
  // 6. Clean up the DOM and memory
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}


async function startLoop() {
    distToStartX = 9999
    distToStartY = 9999
    activeButton.x = 700;
    activeButton.y = 300;
    activeButton.width = 100;
    activeButton.height = 50;
    activeButton.color = dotColorTwo;
    activeButton.label = "START";
    while (!(isClicked == 1 && Math.abs(distToStartX) <= 50 && Math.abs(distToStartY) <= 25)) {
        console.log(["before start data:", isClicked, distToStartX, distToStartY]);
        renderTheFrame();
        distToStartX = currentX - 700;
        distToStartY = currentY - 300;
        await sleep(1000 / 60);
    }
    activeButton.x = getRandomInt(200, 1200);
    activeButton.y = getRandomInt(200, 400);
    activeButton.width = 50;
    activeButton.height = 50;
    activeButton.color = dotColor;
    activeButton.label = String(buttonsClicked + 1);
    mainLoop();
}

async function mainLoop() {
  while(buttonsClicked < 5) {
    renderTheFrame();
    console.log("X:", currentX);
    console.log("Y: ", currentY);
    console.log("isClicked: ", isClicked);
    distToButtonX = currentX - buttonX;
    distToButtonY = currentY - buttonY;
    console.log("DistX: ", distToButtonX);
    console.log("DistY :", distToButtonY);
    data.push({
        time: performance.now(),
        coords: [distToButtonX, distToButtonY, isClicked]
    });
    console.log(data);
    if (isClicked == 1 && isHover(activeButton)) {
        while (isClicked == 1) {
            await sleep(1000 / 60);
            distToButtonX = currentX - buttonX;
            distToButtonY = currentY - buttonY;
            data.push({
                time: performance.now(),
                coords: [distToButtonX, distToButtonY, isClicked]
            });
        }
        buttonsClicked += 1;
        activeButton.x = getRandomInt(200, 1200);
        activeButton.y = getRandomInt(200, 400);
        activeButton.width = 50;
        activeButton.height = 50;
        activeButton.color = dotColor;
        activeButton.label = String(buttonsClicked + 1);
    }
    await sleep(1000 / 60);
  }
  downloadToFile(JSON.stringify(data), "data.json", "application/json")
}
startLoop();
