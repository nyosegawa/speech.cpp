#ifndef SPEECH_H
#define SPEECH_H

#include <stddef.h>
#include <stdint.h>

/*
 * The C API of speech.cpp: speech synthesis with Qwen3-TTS and Irodori-TTS on ggml.
 *
 * Every string passed in or returned is UTF-8, paths included. A function that can fail returns a
 * speech_status, and speech_last_error() then gives the message. No pointer the library returns is ever freed
 * by the caller except through the function named for it. Nothing the caller passes in is kept after the
 * call returns, unless the declaration says otherwise.
 *
 * Threads: a model serves one request at a time, so speech_synthesize() calls on the same model from several
 * threads run one after another; speech_cancel() and the speech_model_* getters may be called from any thread
 * at any time while the model exists. Different models are independent of each other. Loading and freeing a
 * model, listing devices and making a voice file may run on any thread.
 *
 * The library sets ggml up for the whole process when it first touches a device: ggml's warnings and errors go
 * to stderr and its other messages are dropped, and on macOS Metal runs without Metal 4's tensor API.
 */

#ifdef _WIN32
#    if defined(SPEECH_BUILD_SHARED)
#        define SPEECH_API __declspec(dllexport)
#    elif defined(SPEECH_SHARED)
#        define SPEECH_API __declspec(dllimport)
#    else
#        define SPEECH_API
#    endif
#else
#    define SPEECH_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** The version of this API, raised whenever a declaration changes in a way an existing caller notices. */
#define SPEECH_API_VERSION 1

/** The SPEECH_API_VERSION the library was built with, for a caller that loads it at run time. */
SPEECH_API int speech_api_version(void);

/** What a call that can fail returns. */
typedef enum speech_status {
    /** The call did what it was asked. */
    SPEECH_OK = 0,
    /** speech_synthesize(): the callback or speech_cancel() stopped the request before it finished. */
    SPEECH_STOPPED = 1,
    /** The call failed; speech_last_error() says why. */
    SPEECH_ERROR = -1
} speech_status;

/**
 * The message of the last call on the calling thread that returned SPEECH_ERROR, naming what failed and what
 * to do. The library owns it, and it stays valid until the next call into the library on the same thread.
 */
SPEECH_API const char * speech_last_error(void);

/** The kind of a device, as ggml reports it. */
typedef enum speech_device_kind {
    SPEECH_DEVICE_CPU = 0,
    SPEECH_DEVICE_GPU = 1,
    /** A GPU that shares the CPU's memory. */
    SPEECH_DEVICE_IGPU = 2,
    /** An accelerator that ggml uses together with the CPU, such as BLAS. */
    SPEECH_DEVICE_ACCEL = 3
} speech_device_kind;

/** A device the library can run on. The strings are the library's and live as long as the process. */
typedef struct speech_device {
    /** The name to give as speech_model_params.device, such as MTL0, Vulkan1 or CPU. */
    const char * name;
    /** What the device is, such as NVIDIA GeForce RTX 2080. */
    const char * description;
    speech_device_kind kind;
    /** The device's memory in bytes, total and free when asked. */
    uint64_t memory_total;
    uint64_t memory_free;
} speech_device;

/** The number of devices the library can run on. */
SPEECH_API size_t speech_device_count(void);

/** Fills `device` with the device at `index`, which is below speech_device_count(). */
SPEECH_API speech_status speech_device_get(size_t index, speech_device * device);

/** A voice given to an Irodori-TTS model: the name requests use and a reference WAVE file or a voice file. */
typedef struct speech_voice_source {
    const char * name;
    const char * path;
} speech_voice_source;

/**
 * What speech_model_load() loads. Start from speech_model_default_params(), so that a field a later version
 * adds keeps its default.
 */
typedef struct speech_model_params {
    /** The model's GGUF file; its general.architecture chooses the family. */
    const char * model_path;
    /** The codec's GGUF file. */
    const char * codec_path;
    /**
     * The device as speech_device.name gives it, "cpu" for the CPU, or "gpu", NULL or "" for the first GPU.
     * A device that does not exist or does not start is an error; the model is never moved to another one.
     */
    const char * device;
    /** Qwen3-TTS: the talker's context in positions (2048 by default, about 160 s of speech). */
    int context;
    /**
     * Irodori-TTS, which needs at least one: the voices, `n_voices` of them. A WAVE file is 48 kHz and at most
     * 120 s; it is encoded with the codec while the model loads. Qwen3-TTS takes none and speaks with its
     * model's speakers.
     */
    const speech_voice_source * voices;
    size_t n_voices;
    /** Irodori-TTS: the sampler's steps, or 0 for the model's default (4 for MeanFlow, 40 for RF). */
    int steps;
} speech_model_params;

/** The defaults: no paths, the first GPU, a context of 2048, no voices and the model's own steps. */
SPEECH_API speech_model_params speech_model_default_params(void);

/** A loaded model, its codec, its voices and the device they run on. */
typedef struct speech_model speech_model;

/**
 * Loads a model and its codec on the device and runs a short synthesis, so that the GPU's kernels are
 * compiled before the first request. On success `*model` is a model that speech_model_free() frees; on
 * failure it is NULL.
 */
SPEECH_API speech_status speech_model_load(const speech_model_params * params, speech_model ** model);

/** Frees a model and its device. No call on the model may be running or follow. NULL is ignored. */
SPEECH_API void speech_model_free(speech_model * model);

/** How a model passes its audio to speech_synthesize()'s callback. */
typedef enum speech_streaming {
    /** Qwen3-TTS: frame by frame from the first frame on, so audio starts before the rest is made. */
    SPEECH_STREAMING_FRAME = 0,
    /** Irodori-TTS: a request's latent is made whole, then decoded and passed window by window. */
    SPEECH_STREAMING_SENTENCE = 1
} speech_streaming;

/*
 * What a model is. The strings belong to the model and live until speech_model_free().
 */

/** The model's general.name, such as Qwen3-TTS-12Hz-0.6B-CustomVoice or Irodori-TTS-v4.1-Small-MF. */
SPEECH_API const char * speech_model_name(const speech_model * model);
/** The family's architecture: "qwen3tts-talker" or "irodori-tts". */
SPEECH_API const char * speech_model_architecture(const speech_model * model);
/** The sample rate of the audio in Hz: 24000 for Qwen3-TTS, 48000 for Irodori-TTS. */
SPEECH_API int speech_model_sample_rate(const speech_model * model);
SPEECH_API speech_streaming speech_model_streaming(const speech_model * model);
/** The number of voices: Qwen3-TTS's speakers in alphabetical order, or Irodori-TTS's voices in the order given. */
SPEECH_API size_t speech_model_voice_count(const speech_model * model);
/** The name of the voice at `index`, or NULL when `index` is not below speech_model_voice_count(). */
SPEECH_API const char * speech_model_voice(const speech_model * model, size_t index);
/** The number of languages the model speaks. */
SPEECH_API size_t speech_model_language_count(const speech_model * model);
/** The language at `index` as a BCP 47 tag, such as "ja", or NULL past the last one. */
SPEECH_API const char * speech_model_language(const speech_model * model, size_t index);
/** Whether a request's language reaches the model (1, Qwen3-TTS) or is only checked against its languages (0). */
SPEECH_API int speech_model_language_selectable(const speech_model * model);
/** The sampler's steps a request runs (Irodori-TTS), or 0 for a family without steps. */
SPEECH_API int speech_model_steps(const speech_model * model);
/** The ggml backend the model runs on, such as MTL0, Vulkan0 or CPU. */
SPEECH_API const char * speech_model_backend(const speech_model * model);

/** One thing to say. */
typedef struct speech_request {
    /** The text. Irodori-TTS takes one sentence of at most 256 tokens; anything longer is an error. */
    const char * text;
    /** One of the model's voices. */
    const char * voice;
    /**
     * A BCP 47 tag of one of the model's languages or a region or script of one ("ja", "ja-JP"), or "auto",
     * NULL or "" to leave the choice to the model. Any other language is an error.
     */
    const char * language;
    /** The seed of the sampling; the same request and seed give the same audio on the same device. */
    uint64_t seed;
} speech_request;

/**
 * Receives the audio of a request as it is made: `n_samples` mono samples at the model's sample rate, nominally
 * within [-1, 1] and not clamped, valid only during the call. Where the synthesis can stop before it has any
 * audio (between the sampler's steps of Irodori-TTS), it is called with `samples` NULL and `n_samples` 0.
 * Returning nonzero stops the request. It runs on the thread that called speech_synthesize() and must not call
 * speech_synthesize() or speech_model_free() on the same model.
 */
typedef int (*speech_audio_callback)(const float * samples, size_t n_samples, void * user_data);

/**
 * Speaks a request, passing its audio to `on_audio` with `user_data` and returning once it is all passed
 * (SPEECH_OK), once the callback or speech_cancel() stopped it (SPEECH_STOPPED), or on an error (SPEECH_ERROR),
 * after which no more audio comes. A stopped request passes no more audio.
 */
SPEECH_API speech_status speech_synthesize(speech_model * model, const speech_request * request,
                                           speech_audio_callback on_audio, void * user_data);

/**
 * Stops the request the model is speaking when it is called, as the callback would by returning nonzero; a
 * request that starts later, waiting ones included, is not affected. Callable from any thread. A call of the
 * callback already under way when it returns is the last one.
 */
SPEECH_API void speech_cancel(speech_model * model);

/**
 * Makes an Irodori-TTS voice file at `voice_path` from the reference WAVE file `wave_path` (48 kHz, at most
 * 120 s): the reference's codec latent, which loads in milliseconds where a WAVE file is encoded at every
 * load. It loads the model and the codec of `params` on its device for the purpose; the voices of `params`
 * are not used. The voice file names the codec and works only with it. On the CPU the latent is the official
 * encoder's to 99 dB SNR; a GPU computes it in less precision.
 */
SPEECH_API speech_status speech_make_voice(const speech_model_params * params, const char * wave_path,
                                           const char * voice_path);

#ifdef __cplusplus
}
#endif

#endif
