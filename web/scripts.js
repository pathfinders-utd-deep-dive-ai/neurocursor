var currentX = 0
var currentY = 0
var isClicked = 0
var buttonX = 0
var buttonY = 0
var buttonsClicked = 0
var data = []

const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");
function drawRect(x, y, width, height) {
    ctx.fillRect(x-(width/2), y-(height/2), width, height);
}

// Taken from Google AI Overview
function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
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


window.addEventListener('pointerrawupdate', e => {
  const rect = canvas.getBoundingClientRect();
  
  // 1. Subtract the canvas's left/top offset on the page
  // 2. Multiply by the ratio of internal canvas width vs layout width to fix scaling
  currentX = (e.clientX - rect.left) * (canvas.width / rect.width);
  currentY = (e.clientY - rect.top) * (canvas.height / rect.height);
}); // Taken from Gemini

window.addEventListener('pointerdown', e => { if (e.buttons === 1) isClicked = 1; }); // Taken from Google AI Mode
window.addEventListener('pointerup', e => { if (e.buttons === 0) isClicked = 0; }); // Taken from Google AI Mode

const sleep = ms => new Promise(res => setTimeout(res, ms)); // Taken from Google AI Mode

async function startLoop() {
    distToStartX = 9999
    distToStartY = 9999
    drawRect(700, 300, 100, 50);
    while (!(isClicked == 1 && Math.abs(distToStartX) <= 50 && Math.abs(distToStartY) <= 25)) {
        console.log(["before start data:", isClicked, distToStartX, distToStartY])
        distToStartX = currentX - 700;
        distToStartY = currentY - 300;
        await sleep(1000 / 60);
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    buttonX = getRandomInt(200, 1200);
    buttonY = getRandomInt(200, 400);
    drawRect(buttonX, buttonY, 50, 50);
    mainLoop();
}

async function mainLoop() {
  while(buttonsClicked < 5) {
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
    if (isClicked == 1 && Math.abs(distToButtonX) < 25 && Math.abs(distToButtonY) < 25) {
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
        buttonX = getRandomInt(200, 1200);
        buttonY = getRandomInt(200, 400);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawRect(buttonX, buttonY, 50, 50);
    }
    await sleep(1000 / 60);
  }
  downloadToFile(JSON.stringify(data), "data.json", "application/json")
}

startLoop();