# FOMA-encoder

## Introduction
You know how LAME Ain't an MP3 Encoder?  Well, FOMA ain't an audio codec!

FOMA is an acronym for Fear Of Missing Audio, and in this case it's also a package utility that leverages the excellent open-source Opus (lossy) & FLAC (lossless) audio codecs from Xiph.Org Foundation.

FOMA is designed for those that lose sleep decideding whether to use lossy, lossless, or hi-res lossless files.  Instead of on-the-fly transcoding or redundant multi-resolution file sets, FOMA assembles an efficient package of files to provide sensible reconstruction at will, without transcoding.

For example, once a hi-res lossless file is converted to a FOMA package, it generates low bitrate lossy (Opus 128 kbps) & compact resolution lossless (FLAC 16 bit 48 kHz) files.  Additionally, a set of residual files are generated to allow reconstruction of standard resolution lossless (FLAC 24 bit 48 kHz) and high resolution lossless (FLAC 24 bit >48 kHz) files.



## What The ?!

- TH = “Thumbnail”
	- Opus, mono, 6 kbps CBR, fs 24 kHz, frame 40 ms

- LB = “Low Bitrate”
	- Opus, 128 kbps CBR, 48 kHz, frame 2.5 ms

- CR = “Compact Resolution”
	- FLAC, 16 bit, fs 48 kHz

- SR = “Standard Resolution”
	- FLAC, 24 bit, fs 48 kHz

- HR = “High Resolution”
	- FLAC, 24 bit, fs > 48 kHz



## FOMA-encoder
`FOMA-encoder` is a Python script that converts an input audio file (ideally hi-res) into a FOMA package.
```
python3 FOMA-encoder.py <input_file_path>

```



## Geeky Stuff for Future Spatial Audio Coolness

- TH chunk info
	- 960 samples/frame
	- 1 chunk = 32 frames = 30,720 samples = 1.28 s

- LB chunk info
	- 120 samples/frame
	- 1 chunk = 512 frames = 61,440 samples = 1.28 s

- CR, SR, & HR  chunk info
	- 4096 samples/block = 85.33 ms block duration
	- 1 chunk = 15 blocks = 61,440 samples = 1.28 s



## License
`FOMA-encoder` is released under the GPL-3.0 license. See the LICENSE file for more details.


## Support
If you encounter any issues, have questions, or would like to contribute to the Spawn project, please reach out!
