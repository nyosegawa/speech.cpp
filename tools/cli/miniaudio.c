// miniaudio's implementation, for the microphone of speech asr --live. Its configuration, capture without decoders,
// encoders or the engine, is the target's compile definitions, so that microphone.cpp declares the same structures.

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
