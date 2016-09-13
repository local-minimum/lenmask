# lenmask

Detecting length (and possibly more) of _C. elegans_ in images.
If you have other things that are more or less rectangular,
they should work fine too.

## Status

The project is in its early experimental phase,
so don't science this just yet.

## Howto use

You could either import the `mask` module of `lenmask` in python and
run `mask.analysis(path)` where `path` is the path to the image in
question.

But you could also run the `mask` module as a script. For this scenario,
please do `python mask.py --help` for further instructions.

## Installation

You need `scipy`, `matplotlib`, `numpy` installed before running. There are ample instructions on this around the net.

The code is compatible with Python 2.7.x as well as Python 3.x
