var colors = {
  red: [255, 0, 0],
  green: [0, 255, 0],
  pink: [255, 0, 255],
  purple: [255, 0, 255],
};

function isMatch(color) {
  return function(r, g, b) {
    // console.log('comparing', r, g, b, 'with color', color)
    return Math.abs(color[0] - r) < 100 &&
           Math.abs(color[1] - g) < 100 &&
           Math.abs(color[2] - b) < 100;
  };
}


var video = document.getElementById('video');
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');

var controlCanvas = document.getElementById('control');
var controlContext = controlCanvas.getContext('2d');

var nrCols = 16;
var colWidth = canvas.width / nrCols;

function drawGuides() {
  context.strokeStyle = 'rgba(255, 255, 255, .2)';
  context.beginPath();
  for (var i = 0; i < nrCols; i++) {
    context.moveTo(i * colWidth, 0);
    context.lineTo(i * colWidth, canvas.height);
  }
  context.stroke();
}

var grid = {};
resetGrid();

function resetGrid() {
  for (var i = 0; i < nrCols; i++) {
    grid[i] = [];
  }
}

function drawGrid() {
  var colWidth = controlCanvas.width / nrCols;
  var radius = colWidth / 3;
  controlContext.clearRect(0, 0, controlCanvas.width, controlCanvas.height);
  for (var i = 0; i < nrCols; i++) {
    var x = i * colWidth + colWidth / 2; // Center of circle
    grid[i].forEach(function(color, index) {
      controlContext.beginPath();
      var y = index * colWidth + colWidth / 2;
      controlContext.strokeStyle = color;
      controlContext.fillStyle = color;
      controlContext.arc(x, y, radius, 0, 2 * Math.PI);
      controlContext.fill();
    });
  }
}

function getColumn(rect) {
  return parseInt((rect.x + rect.width / 2) / colWidth);
}


tracking.ColorTracker.registerColor('red', isMatch('red'));
tracking.ColorTracker.registerColor('green', isMatch('green'));
// tracking.ColorTracker.registerColor('green', function(r, g, b) {
//   if (r < 50 && g > 200 && b < 50) {
//     return true;
//   }
//   return false;
// });
tracking.ColorTracker.registerColor('pink', isMatch('pink'));
tracking.ColorTracker.registerColor('purple', function(r, g, b) {
        var dx = r - 120;
        var dy = g - 60;
        var dz = b - 210;
        if ((b - g) >= 100 && (r - g) >= 60) {
          return true;
        }
        return dx * dx + dy * dy + dz * dz < 3500;
      });
tracking.ColorTracker.registerColor('green', function(r, g, b) {
  if (r < 50 && g > 200 && b < 50) {
    return true;
  }
  return false;
});
var tracker = new tracking.ColorTracker(['yellow', 'magenta']);

tracking.track('#video', tracker, { camera: true });

tracker.on('track', function(event) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  resetGrid();
  event.data.forEach(function(rect) {
    var color = rect.color;
    if (rect.color === 'custom') {
      rect.color = tracker.customColor;
    }

    context.strokeStyle = rect.color;
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    // context.font = '11px Helvetica';
    // context.fillStyle = "#fff";
    var column = getColumn(rect);
    grid[column].push(color);
    // context.fillText(column, rect.x + rect.width + 5, rect.y + 11);
  });
  drawGuides();
  drawGrid();
});



var tempo = 120;
function initSound() {
  ion.sound({
    sounds: [
      {name: "bd"},
      {name: "ch"},
    ],
    volume: 0.5,
    path: "audio/",
    preload: true,
    multiplay: true,
  });
}

var colorToSound = {
  'yellow': 'bd',
};

function play() {
  var bps = tempo / 60;
  var spb = 1 / bps;
  var interval = spb / 4 * 1000; // From quarter-note to sixteenth
  var i = 0;
  window.setInterval(playNextNote, interval);
  function playNextNote() {
    i >= nrCols && (i = 0);
    grid[i].forEach(function(color, index) {
      var sound = colorToSound[color];
      ion.sound.play(sound);
    });
    i++;
  }
}
initSound();
play();
