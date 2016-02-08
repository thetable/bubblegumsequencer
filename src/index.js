
var Vision = require('./vision.js')
var Sound = require('./sound.js');

var grid = new Array(16);
var vision = new Vision(
  tracking,
  grid,
  document.getElementById('video'),
  document.getElementById('canvas'),
  document.getElementById('control')
);
var sound = new Sound(grid);
sound.play(120);
