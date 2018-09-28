# use with `sed -nf THIS_SCRIPT`

# e.g. f1.png -> f001.png
s/\(f\([0-9]\)\.png\)/\1 f00\2\.png/p
# e.g. f11.png -> f011.png
s/\(f\([0-9][0-9]\)\.png\)/\1 f0\2\.png/p
