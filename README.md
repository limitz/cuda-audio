# CUDA Audio Convolution Reverb
Real-time convolution reverb using impulse response audio files. 
See settings.txt for configuration

### Wave files
This repo is using git-lfs for the storage of binary files(wav impulse responses). The following should automatically download the files during cloning.

```
apt install git-lfs
git lfs install
git clone git@github.com:limitz/cuda-audio
```

### Prerequisites
* CUDA, tested on 10.2 (JetPack 4.6) and 11.something
* CUFFT, should be included with CUDA
* JACK, Jack daemon and development files.
* External Audio Interface (I have a focusrite scarlett)
* USB Midi controller

I've had some issues with jack install and I believe I have jack2 running on the Jetson, and jack1 on my laptop. Don't get discouraged if you find yourself rebooting your system a few times and tweaking parameters while dodging error messages. 

On Jetson nano I set Sample Rate: 44100 Frames/Period: 256 Periods/Buffer: 2. _system still under test_


### Jack on jetson
In order to properly run jack on the Jetson, you may want to modify and rebuild the L4T kernel. I've done this using the scripts found here: https://github.com/JetsonHacksNano/buildKernelAndModules

After downloading the sources, use editConfig to go into the config menu and enable ALSA SEQ support (under sound drivers somewhere).

### Jetson scripts
There are some convenience scripts in the jetson folder.
floorit is a *MUST* before running the application.

* scripts/jfloorit - enable maximum performance
* scripts/chillout - disable maximum performance
* scripts/ghostme - enable X/OpenGL applications without display attached

