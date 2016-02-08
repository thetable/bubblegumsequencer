var path = require('path');
module.exports = {
  entry: {
    app: [
      './src/index.js',
      'ion-sound',
      'tracking/build/tracking.js',
      'tinycolor2',
    ]
  },
  output: {
    path: __dirname,
    filename: 'bubblegum.js',
  },
}
