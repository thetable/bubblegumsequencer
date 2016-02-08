jest.dontMock('../vision.js');
jest.dontMock('tinycolor2');

var mockTracking = {
  track: function() {},
  ColorTracker: function() {
    return {
      on: function() {},
    }
  }
};

describe('Vision', function() {
  var vision, grid;

  beforeEach(function() {
    var video = {id: 'video'};
    grid = new Array(16);
    var Vision = require('../vision.js');
    vision = new Vision(mockTracking, grid, video);
  });


  describe('resetGrid', function() {
    it('resets to empty arrays', function() {
      expect(grid.length).toEqual(16);
      expect(grid[0]).toEqual([]);
      grid[1] = [1,1,1];
      vision.resetGrid();
      expect(grid[0]).toEqual([]);
    });
  });

  describe('getMatcher', function() {
    describe('for red', function() {
      it('should match 255,0,0', function() {
        var f = vision.getMatcher(255, 0, 0);
        expect(f(255, 0, 0)).toBeTruthy();
      });

      it('should match 245,0,0', function() {
        var f = vision.getMatcher(255, 0, 0);
        expect(f(255, 0, 0)).toBeTruthy();
      });

      it('should not match blue', function() {
        var f = vision.getMatcher(255, 0, 0);
        expect(f(0, 0, 255)).toBeFalsy();
      })
    });
  });

  describe('hueDiff', function() {
    it('should compute correct angle between two hues', function() {
      expect(vision.hueDiff(0, 360)).toEqual(0);
      expect(vision.hueDiff(350, 10)).toEqual(20);
      expect(vision.hueDiff(90, 270)).toEqual(180);
      expect(vision.hueDiff(90, 180)).toEqual(90);
      expect(vision.hueDiff(0, 0)).toEqual(0);
      expect(vision.hueDiff(360, 360)).toEqual(0);
    });
  });
});
