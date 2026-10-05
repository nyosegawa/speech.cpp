#ifndef SPEECH_H
#define SPEECH_H

#include <stddef.h>
#include <stdint.h>

/*
 * The C API of speech.cpp on ggml: speech synthesis with Qwen3-TTS and Irodori-TTS, and speech recognition with
 * FastConformer. A model does one of the two (speech_model_task()): speech_synthesize() runs a synthesis model and
 * speech_transcribe() a recognition model, and either function given a model of the other task is an error.
 *
 * Every string passed in or returned is UTF-8, paths included. A function that can fail returns a
 * speech_status, and speech_last_error() then gives the message. No pointer the library returns is ever freed
 * by the caller except through the function named for it. Nothing the caller passes in is kept after the
 * call returns, unless the declaration says otherwise.
 *
 * Threads: a model serves one request at a time, so speech_synthesize() or speech_transcribe() calls on the same
 * model from several threads run one after another; speech_cancel() and the speech_model_* getters may be called
 * from any thread at any time while the model exists. Different models are independent of each other. Loading and
 * freeing a model, listing devices and making a voice file may run on any thread.
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

/**
 * The version of this API, raised whenever a declaration changes in a way an existing caller notices. An added
 * function does not raise it, so a library of an older release with the same API version may lack a function
 * this header declares.
 */
#define SPEECH_API_VERSION 3

/** The SPEECH_API_VERSION the library was built with, for a caller that loads it at run time. */
SPEECH_API int speech_api_version(void);

/**
 * The release of speech.cpp the library was built from, such as "0.4.0": MAJOR.MINOR.PATCH under Semantic
 * Versioning, the release's tag without its "v". The string is the library's and lives as long as the process.
 */
SPEECH_API const char * speech_version(void);

/** What a call that can fail returns. */
typedef enum speech_status {
    /** The call did what it was asked. */
    SPEECH_OK = 0,
    /** speech_synthesize() or speech_transcribe(): the callback or speech_cancel() stopped the request. */
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

/** What a model does. */
typedef enum speech_task {
    /** Text to audio, through speech_synthesize(): Qwen3-TTS and Irodori-TTS. */
    SPEECH_TASK_SYNTHESIS = 0,
    /** Audio to text, through speech_transcribe(): FastConformer. */
    SPEECH_TASK_RECOGNITION = 1
} speech_task;

/** A voice given to an Irodori-TTS model: the name requests use and a reference WAVE file or a voice file. */
typedef struct speech_voice_source {
    const char * name;
    const char * path;
} speech_voice_source;

/**
 * What speech_model_load() loads. Start from speech_model_default_params(), so that a field a later version
 * adds keeps its default. A family takes only the fields it has a use for: a field it does not take, given a value
 * other than its default, is an error that names the field.
 */
typedef struct speech_model_params {
    /**
     * The model's GGUF file, which holds the whole model, its codec included; its general.architecture chooses the
     * family. A file converted for a release before 0.7.0, or of a layout newer than this library reads, is an error
     * that says so.
     */
    const char * model_path;
    /**
     * The device as speech_device.name gives it, "cpu" for the CPU, or "gpu", NULL or "" for the first GPU.
     * A device that does not exist or does not start is an error; the model is never moved to another one.
     */
    const char * device;
    /**
     * Irodori-TTS, which needs at least one: the voices, `n_voices` of them. A WAVE file is at most 120 s at any
     * rate; it is resampled to 48 kHz and encoded with the codec while the model loads. Qwen3-TTS takes none and
     * speaks with its model's speakers, and a recognition model takes none.
     */
    const speech_voice_source * voices;
    size_t n_voices;
    /**
     * Irodori-TTS: the sampler's steps, or 0 for the model's default (4 for MeanFlow, 40 for RF). The other families
     * have no steps and take only 0.
     */
    int steps;
} speech_model_params;

/** The defaults: no path, the first GPU, no voices and the model's own steps. */
SPEECH_API speech_model_params speech_model_default_params(void);

/** A loaded model, with its voices where it has them, and the device it runs on. */
typedef struct speech_model speech_model;

/**
 * Loads a model from its GGUF file on the device, and runs a short synthesis or recognition, so that the GPU's
 * kernels are compiled before the first request. general.architecture of the model's GGUF chooses the family. On
 * success `*model` is a model that speech_model_free() frees; on failure it is NULL.
 */
SPEECH_API speech_status speech_model_load(const speech_model_params * params, speech_model ** model);

/** Frees a model and its device. No call on the model may be running or follow. NULL is ignored. */
SPEECH_API void speech_model_free(speech_model * model);

/** How a model passes its audio to speech_synthesize()'s callback. */
typedef enum speech_streaming {
    /** Qwen3-TTS: frame by frame from the first frame on, so audio starts before the rest is made. */
    SPEECH_STREAMING_FRAME = 0,
    /** Irodori-TTS: a request's latent is made whole, then decoded and passed window by window. */
    SPEECH_STREAMING_SENTENCE = 1,
    /** A recognition model, which makes no audio. */
    SPEECH_STREAMING_NONE = 2
} speech_streaming;

/*
 * What a model is. The strings belong to the model and live until speech_model_free().
 */

/**
 * The model's general.name, such as Qwen3-TTS-12Hz-0.6B-CustomVoice, Irodori-TTS-v4.1-Small-MF or
 * parakeet-tdt_ctc-0.6b-ja.
 */
SPEECH_API const char * speech_model_name(const speech_model * model);
/** The family's architecture: "qwen3-tts", "irodori-tts" or "fastconformer". */
SPEECH_API const char * speech_model_architecture(const speech_model * model);
/** Whether the model speaks (speech_synthesize()) or recognizes speech (speech_transcribe()). */
SPEECH_API speech_task speech_model_task(const speech_model * model);
/**
 * The sample rate in Hz of the audio a synthesis model makes (24000 for Qwen3-TTS, 48000 for Irodori-TTS) or a
 * recognition model recognizes (16000 for FastConformer). Audio given at another rate, to recognize or as a voice's
 * reference, is resampled to it.
 */
SPEECH_API int speech_model_sample_rate(const speech_model * model);
/** How a synthesis model streams its audio; SPEECH_STREAMING_NONE for a recognition model. */
SPEECH_API speech_streaming speech_model_streaming(const speech_model * model);
/**
 * The number of voices: Qwen3-TTS's speakers in alphabetical order, or Irodori-TTS's voices in the order given. A
 * recognition model has none.
 */
SPEECH_API size_t speech_model_voice_count(const speech_model * model);
/** The name of the voice at `index`, or NULL when `index` is not below speech_model_voice_count(). */
SPEECH_API const char * speech_model_voice(const speech_model * model, size_t index);
/** The number of languages the model speaks or recognizes. */
SPEECH_API size_t speech_model_language_count(const speech_model * model);
/** The language at `index` as a BCP 47 tag, such as "ja", or NULL past the last one. */
SPEECH_API const char * speech_model_language(const speech_model * model, size_t index);
/**
 * Whether a request's language reaches the model (1, Qwen3-TTS) or is only checked against its languages (0,
 * Irodori-TTS and FastConformer).
 */
SPEECH_API int speech_model_language_selectable(const speech_model * model);
/** The sampler's steps a request runs (Irodori-TTS), or 0 for a family without steps, recognition models among them. */
SPEECH_API int speech_model_steps(const speech_model * model);
/** The ggml backend the model runs on, such as MTL0, Vulkan0 or CPU. */
SPEECH_API const char * speech_model_backend(const speech_model * model);

/**
 * One thing to say. Start from speech_request_default(), so that a field a later version adds keeps its default.
 * A request that asks a model for what it cannot do is an error; nothing is ignored.
 */
typedef struct speech_request {
    /**
     * The text. Irodori-TTS takes one sentence of at most 256 tokens, and Qwen3-TTS what leaves its talker room for its
     * longest speech (24565 tokens); anything longer is an error.
     */
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
    /**
     * The speaking rate against the model's own, 1 by default. Irodori-TTS takes 0.25 to 4 and divides the length
     * by it, whether fixed by `seconds` or predicted and scaled by `duration_scale`, as Irodori-TTS-Server turns
     * OpenAI's speed into them. Qwen3-TTS has no control of its rate and refuses any speed but 1.
     */
    double speed;
    /**
     * Irodori-TTS: the length of the speech in seconds, or 0 (the default) for the length its duration predictor
     * gives. Divided by `speed`, it must lie within 0.5 to 30 s. The audio is at most this long and ends earlier
     * where the speech falls silent. Qwen3-TTS refuses any length.
     */
    double seconds;
    /**
     * Irodori-TTS: the factor of the predicted length, above 0 and 1 by default; the scaled length is kept within
     * 0.5 to 30 s. It cannot be given together with `seconds`. Qwen3-TTS refuses any factor but 1.
     */
    double duration_scale;
} speech_request;

/** The defaults: no text or voice, the model's choice of language, seed 0, speed 1 and the predicted length. */
SPEECH_API speech_request speech_request_default(void);

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
 * after which no more audio comes. A stopped request passes no more audio. A recognition model is an error.
 */
SPEECH_API speech_status speech_synthesize(speech_model * model, const speech_request * request,
                                           speech_audio_callback on_audio, void * user_data);

/**
 * One piece of audio to recognize. Start from speech_transcription_request_default(), so that a field a later version
 * adds keeps its default. A request that asks a model for what it cannot do is an error; nothing is ignored.
 */
typedef struct speech_transcription_request {
    /** `n_samples` mono samples, nominally within [-1, 1]; at least one is needed. */
    const float * samples;
    size_t n_samples;
    /**
     * The rate of the samples in Hz, any positive rate. Audio at a rate other than the model's,
     * speech_model_sample_rate(), is resampled to it with torchaudio's windowed sinc at librosa's kaiser_best
     * settings. Two rates whose ratio in lowest terms has a term above 4096 (44101 Hz and 16000 Hz) are an error.
     */
    int sample_rate;
    /**
     * A BCP 47 tag of one of the model's languages or a region or script of one ("ja", "ja-JP"), or "auto", NULL or
     * "" to leave the choice to the model. Any other language is an error.
     */
    const char * language;
} speech_transcription_request;

/** The defaults: no samples, a sample rate of 0 (to be set) and the model's choice of language. */
SPEECH_API speech_transcription_request speech_transcription_request_default(void);

/**
 * Receives the text of a request: UTF-8, NUL-terminated, valid only during the call. FastConformer calls it once,
 * with the text of the whole audio. Returning nonzero stops the request, which then returns SPEECH_STOPPED and passes
 * no more text. It runs on the thread that called speech_transcribe() and must not call speech_transcribe() or
 * speech_model_free() on the same model.
 */
typedef int (*speech_text_callback)(const char * text, void * user_data);

/**
 * Recognizes the speech in a request's audio, passing its text to `on_text` with `user_data`, and returns once the
 * text is passed (SPEECH_OK), once the callback or speech_cancel() stopped it (SPEECH_STOPPED), or on an error
 * (SPEECH_ERROR). The audio is checked before any work starts: no samples, a rate the library cannot resample from or a
 * language it does not recognize is an error. A synthesis model is an error. FastConformer recognizes the whole audio
 * in one pass, so speech_cancel() takes effect before the encoder starts or once it has run; its time and memory grow
 * with the square of the audio's length for a model that attends over the whole audio (parakeet), and with its length
 * for one that attends locally (reazonspeech-nemo-v2).
 */
SPEECH_API speech_status speech_transcribe(speech_model * model, const speech_transcription_request * request,
                                           speech_text_callback on_text, void * user_data);

/**
 * Stops the request the model is speaking or recognizing when it is called, as the callback would by returning
 * nonzero; a request that starts later, waiting ones included, is not affected. Callable from any thread. A call of
 * the callback already under way when it returns is the last one.
 */
SPEECH_API void speech_cancel(speech_model * model);

/**
 * Makes an Irodori-TTS voice file at `voice_path` from the reference WAVE file `wave_path` (at most 120 s at any rate,
 * resampled to 48 kHz): the reference's codec latent, which loads in milliseconds where a WAVE file is encoded at every
 * load. It loads the model of `params` on its device for the purpose; the voices of `params` are not used. The voice
 * file carries the hash of the codec inside the model file and works with every model file of the same codec, in any
 * type. On the CPU the latent is the official encoder's to 99 dB SNR; a GPU computes it in less precision.
 */
SPEECH_API speech_status speech_make_voice(const speech_model_params * params, const char * wave_path,
                                           const char * voice_path);

#ifdef __cplusplus
}
#endif

#endif
