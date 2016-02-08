var path = require('path');
module.exports = {
  entry: {
    app: ['./src/index.js', 'ion-sound', 'tracking/build/tracking.js']
  },
  output: {
    path: __dirname,
    filename: 'bubblegum.js',
  },
}
