#pragma once

#include <memory>

#include "httplib.h"

#include "served-models.h"

// POST /v1/audio/enhancements: speech.cpp's own endpoint, which OpenAI's audio API does not have, on the enhancement
// model held: its request is a create transcription's form, and its answer a create speech's audio, at the rate the
// audio was enhanced at. A wav waits for the whole audio; pcm sends the audio as it is enhanced, from the first chunk's on.

namespace server {

/**
 * Answers an enhancement request on `served`: the library's refusal of a value at once, a refusal of the audio before
 * any of it is sent, the enhanced audio as wav or pcm, and the request stopped for a client that goes away while it waits
 * or runs.
 */
void answer_enhancement(std::shared_ptr<Served> served, const httplib::Request & req, httplib::Response & res);

}  // namespace server
