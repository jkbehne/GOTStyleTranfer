# GOTStyleTransfer

## Main script usage
A typical use of this program involves going to the directory of the script and running:

python StyleTransfer.py --contentPath images/Buildings.jpg --stylePath images/StarryNight.jpg --size 512 --outName Example.png --fineToCoarse

When  this is run, the output file will be place by default to samples/Example.png. Note the size of 512 is just for conveniece. Larger values can be used if desired, but the process can become substantially slower. Also note the --fineToCoarse flag usually results in more visually pleasing outputs, but is not required. There is a flag for GPU support (--cuda), but there is currently no guarantee that this will work, as GPU support is only required to perform style transfer on very large images. Otherwise, it generally takes under 30 seconds to run the style transfer with only CPUs.
