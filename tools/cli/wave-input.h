#pragma once

#include <string>

#include "error.h"
#include "failure.h"
#include "speech.h"
#include "wav.h"

// The WAVE files that speech asr and speech vad read.

/**
 * The samples of a WAVE file. A file that cannot be read is an io failure, and one whose content is not WAVE audio
 * speech.cpp reads is the audio's invalid_argument, as the library calls audio it cannot take.
 */
inline Wav read_audio(const std::string & path) {
    try {
        return read_wav(path);
    } catch (const Error & e) {
        throw Failure(speech_status_name(e.fault() == Fault::Io ? SPEECH_ERROR_IO : SPEECH_ERROR_INVALID_ARGUMENT), "audio", e.what());
    }
}
