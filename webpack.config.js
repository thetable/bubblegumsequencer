var path = require('path');
module.exports = {
  entry: './src/index.js',
  output: {
    path: __dirname,
    filename: 'bubblegum.js',
  },
  loaders: [
    {
      test: path.join(__dirname, 'src'),
      loader: 'babel-loader',
    }
  ],
}
