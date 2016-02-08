jest.dontMock('../sound.js');

describe('Sound', function() {
  it('instantiates', function() {
    var grid = new Array(16);
    var Sound = require('../sound.js')
    var sound = new Sound(grid);
  });
});
