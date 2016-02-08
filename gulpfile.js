var gulp = require('gulp');
var serve = require('gulp-serve');
var nodeunit = require('gulp-nodeunit');

gulp.task('serve', serve('.'));

gulp.task('test', function() {
  return gulp.src('test/test.js')
    .pipe(nodeunit());
});
