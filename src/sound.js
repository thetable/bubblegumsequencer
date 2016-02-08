function Sound(grid) {
  var colorToSound = {
    'yellow': 'bd',
  };

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

  this.play = function(tempo) {
    var bps = tempo / 60; // Beats per second
    var spb = 1 / bps; // Seconds-per-beat
    var interval = spb / 4 * 1000; // From quarter-note to sixteenth
    var i = 0;
    window.setInterval(playNextNote, interval);
    function playNextNote() {
      i >= grid.length && (i = 0);
      grid[i].forEach(function(color, index) {
        var sound = colorToSound[color];
        ion.sound.play(sound);
      });
      i++;
    }
  }
}

module.exports = Sound;
