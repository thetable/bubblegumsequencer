var Vision = require('./vision.js')

var vision = new Vision(
  document.getElementById('video'),
  document.getElementById('canvas'),
  document.getElementById('control')
);

var tempo = 120;
function initSound() {
  ion.sound({
    sounds: [
      // {name: "bd"},
      // {name: "ch"},
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
// initSound();
// play();
