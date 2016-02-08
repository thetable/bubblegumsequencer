function Vision(video, videoCanvas, controlCanvas) {
  var NR_COLS = 16;
  var grid = {};
  var tracker = new tracking.ColorTracker(['yellow', 'magenta']);
  tracking.track('#' + video.id, tracker, { camera: true });
  tracker.on('track', onTrack);

  resetGrid();

  function onTrack(event) {
    var context = videoCanvas.getContext('2d');
    context.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
    resetGrid();
    event.data.forEach(function(rect) {
      var color = rect.color;
      if (rect.color === 'custom') {
        rect.color = tracker.customColor;
      }

      context.strokeStyle = rect.color;
      context.strokeRect(rect.x, rect.y, rect.width, rect.height);
      var column = getColumn(rect);
      grid[column].push(color);
    });
    drawGuides();
    visualizeGridState();
  }

  function resetGrid() {
    for (var i = 0; i < NR_COLS; i++) {
      grid[i] = [];
    }
  }

  function getColumn(rect) {
    var colWidth = videoCanvas.width / NR_COLS;
    return parseInt((rect.x + rect.width / 2) / colWidth);
  }

  /*
   * Draw helper lines on video canvas to help user position camera and holes correctly.
   */
  function drawGuides() {
    var context = videoCanvas.getContext('2d');
    var colWidth = videoCanvas.width / NR_COLS;
    context.strokeStyle = 'rgba(255, 255, 255, .2)';
    context.beginPath();
    for (var i = 0; i < NR_COLS; i++) {
      context.moveTo(i * colWidth, 0);
      context.lineTo(i * colWidth, canvas.height);
    }
    context.stroke();
  }

  /*
   * Visualize state of grid in secondary canvas.
   */
  function visualizeGridState() {
    var colWidth = controlCanvas.width / NR_COLS;
    var radius = colWidth / 3;
    var controlContext = controlCanvas.getContext('2d');
    controlContext.clearRect(0, 0, controlCanvas.width, controlCanvas.height);
    for (var i = 0; i < NR_COLS; i++) {
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
}

module.exports = Vision;
