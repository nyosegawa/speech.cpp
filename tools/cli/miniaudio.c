// miniaudio's implementation, for the microphone of speech asr --live and the microphone and audio output of speech talk
// --live. Its configuration, capture and playback without decoders, encoders or the engine, is the target's compile
// definitions, so that microphone.cpp and speaker.cpp declare the same structures.

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
