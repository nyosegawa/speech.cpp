#ifndef SPEECH_H
#define SPEECH_H

#include <stddef.h>
#include <stdint.h>

/*
 * The C API of speech.cpp on ggml: speech synthesis and speech recognition with the models of the families the
 * library runs. A model does one task (speech_model_info_task()): a synthesis model turns the text of a request into
 * audio through speech_synthesize(), and a recognition model turns the audio of a request into text through
 * speech_transcribe().
 *
 * What a request may ask of a model is one vocabulary of options (speech_option), each with a fixed name that the
 * worker, the HTTP server and the command line use verbatim. Each family declares which options it takes, with their
 * type, default and range or choices, and the library checks every value a caller sets against that declaration as
 * it is set. The same declaration is readable before the model is loaded, through speech_model_info_open(), so a
 * caller learns what a model takes without trying. An option a model does not take is an error, except at the
 * option's neutral value, which every model accepts.
 *
 * Every string passed in or returned is UTF-8, paths included. A function that can fail returns a speech_status; on
 * failure speech_last_error() gives the message and speech_last_error_option() the input it concerns. Every object
 * the library returns is freed only through the function named for it, and every string it returns lives as long as
 * the object it was read from unless the declaration says otherwise. Nothing the caller passes in is kept after the
 * call returns.
 *
 * Threads: a model serves one request at a time, so requests that several threads start on the same model run one
 * after another. A request is used by one thread at a time, except for speech_request_cancel(), which any thread may
 * call while the request exists. Model information never changes once made and may be read from any thread.
 * Different models are independent of each other.
 *
 * The library checks what it is given before ggml sees it, so that no input a caller passes makes ggml abort the
 * process.
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
 * The major version of this API. It rises when a declaration changes in a way an existing caller notices, and a
 * program built against one major version runs only against a library of the same major version. The shared
 * library's SOVERSION is this number.
 */
#define SPEECH_API_VERSION_MAJOR 3

/**
 * The minor version of this API. It rises when a function, an option or an enum value is added, and returns to 0
 * when the major version rises. A program built against minor version m runs against a library of the same major
 * version and a minor version of m or more.
 */
#define SPEECH_API_VERSION_MINOR 0

/** The SPEECH_API_VERSION_MAJOR the library was built with, for a caller that loads it at run time. */
SPEECH_API int speech_api_version_major(void);

/** The SPEECH_API_VERSION_MINOR the library was built with. */
SPEECH_API int speech_api_version_minor(void);

/**
 * The release of speech.cpp the library was built from, such as "0.7.0": MAJOR.MINOR.PATCH under Semantic
 * Versioning, the release's tag without its "v". The string lives as long as the process.
 */
SPEECH_API const char * speech_version(void);

/** What a call that can fail returns. A negative value is an error. */
typedef enum speech_status {
    /** The call did what it was asked. */
    SPEECH_OK = 0,
    /** A request was stopped by speech_request_cancel() or by one of its callbacks returning nonzero. */
    SPEECH_CANCELLED = 1,
    /** The caller's mistake: a NULL pointer, an empty text, a value of the wrong type, a call out of order. */
    SPEECH_ERROR_INVALID_ARGUMENT = -1,
    /** The model cannot do what was asked: an option it does not take, given a value other than the neutral one. */
    SPEECH_ERROR_UNSUPPORTED = -2,
    /** The model takes the option, but not this value: outside its range, not one of its choices, or too long. */
    SPEECH_ERROR_OUT_OF_RANGE = -3,
    /**
     * A model or voice file that cannot be used: not GGUF, an architecture no family has, a layout this release does
     * not read, or a key or tensor that is missing, of the wrong type, or with a value the layout does not allow.
     */
    SPEECH_ERROR_MODEL_FILE = -4,
    /** A device that is not there, does not start, or fails while it computes. */
    SPEECH_ERROR_DEVICE = -5,
    /** The memory of the host or of the device ran out. */
    SPEECH_ERROR_OUT_OF_MEMORY = -6,
    /** A file that cannot be opened, read or written. */
    SPEECH_ERROR_IO = -7,
    /** A defect of the library. */
    SPEECH_ERROR_INTERNAL = -8
} speech_status;

/**
 * The name of a status in snake_case, as the worker, the HTTP server and the command line write it: "ok",
 * "cancelled", "invalid_argument", "unsupported", "out_of_range", "model_file", "device", "out_of_memory", "io" or
 * "internal". NULL for a value the library does not know. The string lives as long as the process.
 */
SPEECH_API const char * speech_status_name(speech_status status);

/**
 * The message of the last call on the calling thread that failed, naming what failed and what to do. It stays valid
 * until the next call into the library on the same thread.
 */
SPEECH_API const char * speech_last_error(void);

/**
 * The input the last failed call on the calling thread concerns: an option's name as speech_option_name() gives it,
 * "text" or "audio" for a request's input, "device", "threads" or "warmup" for a load parameter, "name" or "path"
 * for a voice being added, or "model_path", "reference_path" or "voice_path" for a voice file being made. NULL when
 * the failure concerns no single input. It stays valid until the next call into the library on the same thread.
 */
SPEECH_API const char * speech_last_error_option(void);

/** How much a log message matters. */
typedef enum speech_log_level {
    SPEECH_LOG_DEBUG = 0,
    SPEECH_LOG_INFO = 1,
    SPEECH_LOG_WARN = 2,
    SPEECH_LOG_ERROR = 3
} speech_log_level;

/**
 * Receives the library's log messages and ggml's. `text` is one line or a piece of one, as ggml writes it, and is
 * valid only during the call. It is called on whichever thread is inside the library, so it must be safe to call
 * from several threads at once.
 */
typedef void (*speech_log_callback)(speech_log_level level, const char * text, void * user_data);

/**
 * Sends every later log message to `callback` with `user_data`, or drops every message when `callback` is NULL. Until
 * it is called, warnings and errors go to stderr and the rest is dropped. It holds for the whole process.
 */
SPEECH_API void speech_log_set(speech_log_callback callback, void * user_data);

/*
 * Devices: the CPU and the GPUs that ggml can run a whole model on, in ggml's order. An accelerator that ggml runs
 * beside the CPU, such as BLAS, cannot run a model by itself and is not listed. The first call that touches a device
 * sets ggml up for the process. On macOS, Metal must run without Metal 4's tensor API, and ggml reads that only from
 * the environment variable GGML_METAL_TENSOR_DISABLE while it lists its devices; the library sets the variable for
 * that listing alone and then restores it, so the host sees the value it had. ggml's Metal backend itself sets
 * AGX_RELAX_CDM_CTXSTORE_TIMEOUT to 1 for the process during that listing, which the library leaves as ggml sets it.
 * Setting the environment is not safe while another thread reads it, so that first call must not run while another
 * thread of the process reads the environment.
 */

/** The kind of a device. */
typedef enum speech_device_kind {
    SPEECH_DEVICE_CPU = 0,
    /** A GPU with memory of its own. */
    SPEECH_DEVICE_GPU = 1,
    /** A GPU that shares the CPU's memory. */
    SPEECH_DEVICE_IGPU = 2
} speech_device_kind;

/** The number of devices the library can run a model on. */
SPEECH_API size_t speech_device_count(void);

/**
 * The name of the device at `index` as a load parameter gives it, such as "MTL0", "Vulkan1" or "CPU", or NULL when
 * `index` is not below speech_device_count(). The string lives as long as the process.
 */
SPEECH_API const char * speech_device_name(size_t index);

/** What the device at `index` is, such as "NVIDIA GeForce RTX 2080", or NULL past the last device. */
SPEECH_API const char * speech_device_description(size_t index);

/** The kind of the device at `index`. An index past the last device is an error. */
SPEECH_API speech_status speech_device_get_kind(size_t index, speech_device_kind * kind);

/** The memory of the device at `index` in bytes: its total, and what is free when it is called. */
SPEECH_API speech_status speech_device_memory(size_t index, uint64_t * total, uint64_t * free_bytes);

/*
 * Options: the vocabulary of what a request may ask, the same for every family. A value is never reused, and a later
 * minor version only appends. Each option has one type. Some have a neutral value that every model accepts, whether
 * or not it takes the option: speed 1, duration_scale 1, language "auto" and timestamps false. The others have none,
 * and a model that does not take one refuses any value of it.
 */
typedef enum speech_option {
    /**
     * "voice", a string: the name of one of the model's voices, compared with case. A synthesis model that takes it
     * requires it.
     */
    SPEECH_OPT_VOICE = 0,
    /**
     * "language", a string: "auto" (the neutral value), or a BCP 47 tag that names one of the model's languages or a
     * region or script of one ("ja", "ja-JP", "zh-Hant"), compared without case. The model's declaration says
     * whether the language steers the model or is only checked against its languages
     * (speech_model_info_option_steers()).
     */
    SPEECH_OPT_LANGUAGE = 1,
    /**
     * "seed", an integer from 0 to 2^53 - 1: the seed of the request's sampling, so that the same request with the
     * same seed on the same device gives the same audio. Without it the library draws one from the same range and
     * reports it in the result.
     */
    SPEECH_OPT_SEED = 2,
    /** "speed", a number: the speaking rate against the model's own, 1 being neutral. */
    SPEECH_OPT_SPEED = 3,
    /** "seconds", a number: the length of the speech in seconds, fixed before the speech is made. */
    SPEECH_OPT_SECONDS = 4,
    /** "duration_scale", a number: the factor of the length the model predicts, 1 being neutral. */
    SPEECH_OPT_DURATION_SCALE = 5,
    /** "steps", an integer: the steps of the model's sampler. */
    SPEECH_OPT_STEPS = 6,
    /**
     * "max_seconds", a number: the longest the speech may be. A synthesis that reaches it stops there and reports
     * SPEECH_STOP_MAX_SECONDS.
     */
    SPEECH_OPT_MAX_SECONDS = 7,
    /**
     * "timestamps", a boolean: whether a recognition's result carries its segments and its tokens with their times,
     * false being neutral.
     */
    SPEECH_OPT_TIMESTAMPS = 8
} speech_option;

/** The type of an option's values, which names the setter that takes them. */
typedef enum speech_type {
    /** UTF-8 text, set with speech_request_set_string(). */
    SPEECH_TYPE_STRING = 0,
    /** A whole number, set with speech_request_set_int(). */
    SPEECH_TYPE_INT = 1,
    /** A finite number, set with speech_request_set_float(). */
    SPEECH_TYPE_FLOAT = 2,
    /** True or false, set with speech_request_set_bool(). */
    SPEECH_TYPE_BOOL = 3
} speech_type;

/** The number of options the library knows; they are the values from 0 to one below it. */
SPEECH_API size_t speech_option_count(void);

/**
 * The fixed name of an option in snake_case, such as "duration_scale", which the command line writes in kebab-case
 * ("--duration-scale"); NULL for a value the library does not know. The string lives as long as the process.
 */
SPEECH_API const char * speech_option_name(speech_option option);

/** The option named `name` in snake_case. A name the library does not know is an error. */
SPEECH_API speech_status speech_option_from_name(const char * name, speech_option * option);

/** The type of an option's values. */
SPEECH_API speech_type speech_option_type(speech_option option);

/*
 * Load parameters: what speech_model_load() and speech_voice_make() take besides the path, the same for every family.
 * NULL in place of a parameters object means the defaults.
 */
typedef struct speech_load_params speech_load_params;

/** Parameters at their defaults: the device "auto", the library's number of threads and no warm-up. */
SPEECH_API speech_status speech_load_params_new(speech_load_params ** params);

/** Frees parameters. NULL is ignored. */
SPEECH_API void speech_load_params_free(speech_load_params * params);

/**
 * The device to run on: "auto" (the default) for the first GPU or integrated GPU the library lists, or the CPU when
 * there is none; "gpu" for the first GPU or integrated GPU; "cpu"; or a device's name as speech_device_name() gives
 * it, compared without case. A device asked for by "gpu" or by name that is not there or does not start is an error
 * when the model loads, and no other device takes its place. The model's information names the device it runs on.
 */
SPEECH_API speech_status speech_load_params_set_device(speech_load_params * params, const char * device);

/**
 * The number of threads the CPU computes with, 1 or more: a preference about how the work runs, which applies to
 * whatever the model computes on the CPU, all of its graphs for a model on the CPU and none of them for a model on a
 * GPU. Without it the library takes the machine's performance cores, or its physical cores where the system does not
 * tell them apart. The model's information reports the number in effect (speech_model_info_threads()).
 */
SPEECH_API speech_status speech_load_params_set_threads(speech_load_params * params, int threads);

/**
 * Whether loading also runs a short request of the model's task (nonzero), so that a GPU's kernels are compiled
 * before the first request rather than during it. Off by default; a program that loads a model to answer requests as
 * they come, such as a server, turns it on.
 */
SPEECH_API speech_status speech_load_params_set_warmup(speech_load_params * params, int warmup);

/** What a model does. */
typedef enum speech_task {
    /** Text to audio, through speech_synthesize(). */
    SPEECH_TASK_SYNTHESIS = 0,
    /** Audio to text, through speech_transcribe(). */
    SPEECH_TASK_RECOGNITION = 1
} speech_task;

/*
 * Model information: what a model file says about its model, read from the metadata without the weights, or taken
 * from a loaded model together with what loading added (the device, the voices added since). Information never
 * changes once made, any thread may read it, and its strings live until speech_model_info_free(). An index past the
 * end gives NULL or 0.
 */
typedef struct speech_model_info speech_model_info;

/**
 * Reads the information of the model file at `path` without loading its weights or touching a device. The file is
 * checked as far as the information needs: its architecture, its layout and the keys the information reads.
 */
SPEECH_API speech_status speech_model_info_open(const char * path, speech_model_info ** info);

/** Frees information. NULL is ignored. */
SPEECH_API void speech_model_info_free(speech_model_info * info);

/** The model's name from its file, such as "Qwen3-TTS-12Hz-0.6B-CustomVoice" or "parakeet-tdt-0.6b-v3". */
SPEECH_API const char * speech_model_info_name(const speech_model_info * info);

/** The architecture of the model's family, the file's general.architecture, such as "irodori-tts". */
SPEECH_API const char * speech_model_info_architecture(const speech_model_info * info);

/** The version of the family's file layout that the file has. */
SPEECH_API uint32_t speech_model_info_layout(const speech_model_info * info);

/** Whether the model speaks or recognizes speech. */
SPEECH_API speech_task speech_model_info_task(const speech_model_info * info);

/**
 * The sample rate in Hz of the audio a synthesis model makes, or of the audio a recognition model recognizes. Audio
 * given at another rate, to recognize or as a voice's reference, is resampled to it.
 */
SPEECH_API int speech_model_info_sample_rate(const speech_model_info * info);

/**
 * Whether a synthesis model passes audio while it is still generating the rest (1), so that its first audio does not
 * wait on the length of the text, or generates a request's whole speech before it decodes the speech into audio (0).
 * 0 for a recognition model.
 */
SPEECH_API int speech_model_info_incremental(const speech_model_info * info);

/** The number of languages the model speaks or recognizes. */
SPEECH_API size_t speech_model_info_language_count(const speech_model_info * info);

/** The language at `index` as a BCP 47 tag, such as "ja". */
SPEECH_API const char * speech_model_info_language(const speech_model_info * info, size_t index);

/**
 * The number of voices: the model's own, then, for information taken from a loaded model, those added with
 * speech_voice_add() in the order they were added.
 */
SPEECH_API size_t speech_model_info_voice_count(const speech_model_info * info);

/** The name of the voice at `index`, which the voice option takes. */
SPEECH_API const char * speech_model_info_voice_name(const speech_model_info * info, size_t index);

/** The language the voice at `index` was recorded in as a BCP 47 tag, or "" when the model file does not say. */
SPEECH_API const char * speech_model_info_voice_language(const speech_model_info * info, size_t index);

/** "female" or "male" for the voice at `index`, or "" when the model file does not say. */
SPEECH_API const char * speech_model_info_voice_gender(const speech_model_info * info, size_t index);

/** The model's description of the voice at `index` in English, or "" when the model file has none. */
SPEECH_API const char * speech_model_info_voice_description(const speech_model_info * info, size_t index);

/**
 * Whether the model takes voices made from reference recordings (1), through speech_voice_add() and
 * speech_voice_make().
 */
SPEECH_API int speech_model_info_voice_files(const speech_model_info * info);

/**
 * The hash a voice file must carry to work with the model, the SHA-256 of its codec's official weights in lowercase
 * hexadecimal, or NULL for a model that takes no voice files.
 */
SPEECH_API const char * speech_model_info_voice_codec(const speech_model_info * info);

/** The most tokens of text a synthesis request may have, in the model's own tokens; 0 for a recognition model. */
SPEECH_API size_t speech_model_info_max_text_tokens(const speech_model_info * info);

/**
 * The number of the model's tokens that `text` takes, counted as a synthesis request counts it against
 * speech_model_info_max_text_tokens(), so that a caller can split a long text before it sends it. A recognition model
 * is SPEECH_ERROR_UNSUPPORTED.
 */
SPEECH_API speech_status speech_model_info_text_tokens(const speech_model_info * info, const char * text,
                                                       size_t * n_tokens);

/** The number of options the model takes. */
SPEECH_API size_t speech_model_info_option_count(const speech_model_info * info);

/** The option at `index` of those the model takes, in the order of the vocabulary. */
SPEECH_API speech_option speech_model_info_option(const speech_model_info * info, size_t index);

/**
 * Whether the model takes `option` (1). An option it does not take is accepted only at the option's neutral value,
 * and refused at any value when the option has none.
 */
SPEECH_API int speech_model_info_takes(const speech_model_info * info, speech_option option);

/** Whether a request to the model must set `option` (1). */
SPEECH_API int speech_model_info_option_required(const speech_model_info * info, speech_option option);

/**
 * Whether the value of `option` steers what the model does (1), or is only checked against the option's choices and
 * then not used (0), as a language is by a model that has one language or finds the language itself.
 */
SPEECH_API int speech_model_info_option_steers(const speech_model_info * info, speech_option option);

/**
 * Whether `option` has a default that the model uses when a request does not set it (1). An option without one is
 * required (voice), drawn by the library when it is not set (seed), or without effect until it is set (seconds,
 * max_seconds).
 */
SPEECH_API int speech_model_info_option_has_default(const speech_model_info * info, speech_option option);

/**
 * The default of a string option. An option the model does not take, an option of another type or one without a
 * default is an error.
 */
SPEECH_API speech_status speech_model_info_option_default_string(const speech_model_info * info, speech_option option,
                                                                 const char ** value);

/** The default of an integer option, under the rules of speech_model_info_option_default_string(). */
SPEECH_API speech_status speech_model_info_option_default_int(const speech_model_info * info, speech_option option,
                                                              int64_t * value);

/** The default of a number option, under the rules of speech_model_info_option_default_string(). */
SPEECH_API speech_status speech_model_info_option_default_float(const speech_model_info * info, speech_option option,
                                                                double * value);

/** The default of a boolean option, under the rules of speech_model_info_option_default_string(). */
SPEECH_API speech_status speech_model_info_option_default_bool(const speech_model_info * info, speech_option option,
                                                               int * value);

/**
 * The range of an integer or number option: `*minimum` and `*maximum`, -INFINITY or INFINITY where the option has no
 * bound, and `*minimum_exclusive` nonzero when the minimum itself lies outside the range. A value within it can still
 * be refused when the request runs, for what it does together with another option or with the text, and the error
 * then names the option. An option the model does not take, or one of another type, is an error.
 */
SPEECH_API speech_status speech_model_info_option_range(const speech_model_info * info, speech_option option,
                                                        double * minimum, double * maximum, int * minimum_exclusive);

/**
 * The number of values a string option takes: the voices for "voice", and the languages for "language", which also
 * takes "auto" and a region or script of each. 0 for an option of another type.
 */
SPEECH_API size_t speech_model_info_option_choice_count(const speech_model_info * info, speech_option option);

/** The value at `index` of those a string option takes. */
SPEECH_API const char * speech_model_info_option_choice(const speech_model_info * info, speech_option option,
                                                        size_t index);

/** The size of the model file in bytes. */
SPEECH_API uint64_t speech_model_info_file_bytes(const speech_model_info * info);

/** The bytes the model's weights take on a device: the sum of its tensors as the file stores them. */
SPEECH_API uint64_t speech_model_info_weight_bytes(const speech_model_info * info);

/**
 * The name of the device the model runs on as speech_device_name() gives it, for information taken from a loaded
 * model, or NULL for information read from a file.
 */
SPEECH_API const char * speech_model_info_device(const speech_model_info * info);

/**
 * The number of threads the CPU computes a loaded model's graphs with: the number the load parameters set, or the
 * library's default, for a model on the CPU, and 0 for a model on a GPU, none of whose graphs runs on the CPU. 0 for
 * information read from a file.
 */
SPEECH_API int speech_model_info_threads(const speech_model_info * info);

/**
 * All of the information as one JSON object: the form the worker's ready message, the HTTP server's model object and
 * `speech info --json` carry, so that every program that shows it shows the same.
 */
SPEECH_API const char * speech_model_info_json(const speech_model_info * info);

/** The number of metadata entries in the model file, for a caller that needs a key the information does not name. */
SPEECH_API size_t speech_model_info_meta_count(const speech_model_info * info);

/** The key of the metadata entry at `index`, such as "general.license". */
SPEECH_API const char * speech_model_info_meta_key(const speech_model_info * info, size_t index);

/**
 * The value of the metadata entry at `index` as JSON text: a string, a number, true or false, or an array of them. A
 * long array, such as a tokenizer's vocabulary, is written whole.
 */
SPEECH_API const char * speech_model_info_meta_value(const speech_model_info * info, size_t index);

/** A loaded model on its device, with its voices. */
typedef struct speech_model speech_model;

/**
 * Loads the model file at `path`, which holds every part of the model, its codec included, on the device that the
 * parameters name, or with the defaults when `params` is NULL. The file's general.architecture chooses the family. On
 * success `*model` is a model that speech_model_free() frees; on failure it is NULL.
 */
SPEECH_API speech_status speech_model_load(const char * path, const speech_load_params * params,
                                           speech_model ** model);

/** Frees a model and its device. Every request made for it must be freed first. NULL is ignored. */
SPEECH_API void speech_model_free(speech_model * model);

/**
 * The information of a loaded model as it is at the call: the file's, with the device the model runs on and the
 * voices added so far. The caller frees it with speech_model_info_free(). Any thread may call it.
 */
SPEECH_API speech_status speech_model_get_info(const speech_model * model, speech_model_info ** info);

/*
 * Voices made from reference recordings, for a model whose information says it takes voice files. A voice file holds
 * the codec's latent of a reference recording, the recording's length and rate, the kind of device that encoded it
 * and the hash of the codec it was encoded with, and it works with every model whose codec has that hash.
 */

/**
 * Adds a voice named `name` to the model, from a voice file or from a reference recording in a WAVE file, which is
 * mixed to mono, resampled to the model's rate and encoded with the model's codec on the model's device. The name is
 * compared with case and must not name a voice the model has. A request that is running finishes first; any thread
 * may call it. A model that takes no voice files is SPEECH_ERROR_UNSUPPORTED, and a voice file made with another
 * codec is SPEECH_ERROR_INVALID_ARGUMENT.
 */
SPEECH_API speech_status speech_voice_add(speech_model * model, const char * name, const char * path);

/**
 * Makes a voice file at `voice_path` from the reference recording in the WAVE file at `reference_path`, for the model
 * file at `model_path`, reading only the codec's encoder from the model file and running it on the device that the
 * parameters name. The recording is mixed to mono, resampled to the model's rate, and may be at most as long as the
 * model's file allows. On the CPU the latent is the official encoder's to 99 dB SNR; a GPU computes it in less
 * precision, and the voice file records which kind of device made it.
 */
SPEECH_API speech_status speech_voice_make(const char * model_path, const char * reference_path,
                                           const char * voice_path, const speech_load_params * params);

/*
 * Requests. A request is made for one model and checks every value as it is set against what that model declares:
 * an option the model does not take, at a value other than the option's neutral one, is SPEECH_ERROR_UNSUPPORTED; a
 * value outside the option's range or choices is SPEECH_ERROR_OUT_OF_RANGE; a setter of another type than the
 * option's is SPEECH_ERROR_INVALID_ARGUMENT. A refused value leaves the request as it was, and setting an option
 * again replaces its value. An option the request does not set takes the model's default. What only the whole request
 * shows (a required option left out, two options that cannot go together, a text longer than the model takes) is
 * refused when the request runs, before any work, and the error names the option or input; such a request is left as
 * it was, to be fixed and run again. A request that has started its work runs once.
 */
typedef struct speech_request speech_request;

/** A request for `model`, with no input and no option set, freed with speech_request_free() before the model. */
SPEECH_API speech_status speech_request_new(speech_model * model, speech_request ** request);

/** Frees a request and its result. It must not be running. NULL is ignored. */
SPEECH_API void speech_request_free(speech_request * request);

/**
 * The text a synthesis request speaks, copied; it may not be empty. A recognition model is
 * SPEECH_ERROR_UNSUPPORTED.
 */
SPEECH_API speech_status speech_request_set_text(speech_request * request, const char * text);

/**
 * The audio a recognition request recognizes: `n_samples` mono samples, nominally within [-1, 1], at `sample_rate`
 * Hz, copied. Audio at a rate other than the model's is resampled to it when the request runs. At least one sample is
 * needed. A synthesis model is SPEECH_ERROR_UNSUPPORTED.
 */
SPEECH_API speech_status speech_request_set_audio(speech_request * request, const float * samples, size_t n_samples,
                                                  int sample_rate);

/** Sets a string option, copied. */
SPEECH_API speech_status speech_request_set_string(speech_request * request, speech_option option, const char * value);

/** Sets an integer option. */
SPEECH_API speech_status speech_request_set_int(speech_request * request, speech_option option, int64_t value);

/** Sets a number option. NaN and the infinities are refused. */
SPEECH_API speech_status speech_request_set_float(speech_request * request, speech_option option, double value);

/** Sets a boolean option, to true when `value` is nonzero. */
SPEECH_API speech_status speech_request_set_bool(speech_request * request, speech_option option, int value);

/**
 * Receives how far a request has come while it does work that passes no audio: `done` rises from 0 to 1 as the
 * family's steps finish (the sampler's steps of a speech made whole before its audio, the stages and the decoding of
 * a recognition). The time between two calls is that of one step, which for the encoder of a long recording can be
 * seconds. Returning nonzero stops the request. It runs on the thread that runs the request.
 */
typedef int (*speech_progress_callback)(double done, void * user_data);

/** The callback that receives the request's progress with `user_data`, or NULL for none, the default. */
SPEECH_API speech_status speech_request_set_progress(speech_request * request, speech_progress_callback on_progress,
                                                     void * user_data);

/**
 * Stops the request at the next point where the work under way can stop: before the next piece of audio, the next
 * step or the next stage. The request then returns SPEECH_CANCELLED. A request cancelled before it runs returns
 * SPEECH_CANCELLED without doing any work, and one that has returned is not affected. Any thread may call it while
 * the request exists, and it affects this request alone.
 */
SPEECH_API void speech_request_cancel(speech_request * request);

/**
 * Receives the audio of a synthesis as it is made: `n_samples` mono samples, at least one, at the model's sample
 * rate, nominally within [-1, 1] and not clamped, valid only during the call. Returning nonzero stops the request. It
 * runs on the thread that called speech_synthesize() and must not call into the library with the same model.
 */
typedef int (*speech_audio_callback)(const float * samples, size_t n_samples, void * user_data);

/**
 * Speaks a request, passing its audio to `on_audio` with `user_data`. It returns SPEECH_OK once the speech has ended
 * or has been stopped by a limit, with the reason in the result; SPEECH_CANCELLED once it was cancelled; or an error,
 * after which no more audio comes. The request is checked as a whole before any work starts. A recognition model is
 * SPEECH_ERROR_UNSUPPORTED.
 */
SPEECH_API speech_status speech_synthesize(speech_request * request, speech_audio_callback on_audio, void * user_data);

/**
 * Recognizes the speech in a request's audio. It returns SPEECH_OK once the result holds the text, SPEECH_CANCELLED
 * once it was cancelled, or an error. The whole audio is recognized at once, and how its time and memory grow with
 * its length depends on the model. A synthesis model is SPEECH_ERROR_UNSUPPORTED.
 */
SPEECH_API speech_status speech_transcribe(speech_request * request);

/** Why a request ended. */
typedef enum speech_stop {
    /** It did all it was asked: the speech came to its end, or the audio was recognized whole. */
    SPEECH_STOP_COMPLETE = 0,
    /** The speech reached the request's max_seconds and was stopped there. */
    SPEECH_STOP_MAX_SECONDS = 1,
    /** The speech reached the longest that the model makes, which its file gives, and was stopped there. */
    SPEECH_STOP_MODEL_LIMIT = 2,
    /** speech_request_cancel() or a callback stopped it. */
    SPEECH_STOP_CANCELLED = 3
} speech_stop;

/**
 * The name of a stop reason in snake_case: "complete", "max_seconds", "model_limit" or "cancelled"; NULL for a value
 * the library does not know. The string lives as long as the process.
 */
SPEECH_API const char * speech_stop_name(speech_stop stop);

/** What a request did, owned by the request. */
typedef struct speech_result speech_result;

/**
 * The result of a request that returned SPEECH_OK or SPEECH_CANCELLED, or NULL before the request has run and after
 * it failed. It lives until speech_request_free().
 */
SPEECH_API const speech_result * speech_request_result(const speech_request * request);

/** Why the request ended. */
SPEECH_API speech_stop speech_result_stop(const speech_result * result);

/** The seed a synthesis used, the request's or the one the library drew; -1 for a recognition. */
SPEECH_API int64_t speech_result_seed(const speech_result * result);

/** The number of samples a synthesis passed to its callback, whose length in seconds is this over the sample rate. */
SPEECH_API uint64_t speech_result_samples(const speech_result * result);

/** The text a recognition found, "" when it heard none or was cancelled; NULL for a synthesis. */
SPEECH_API const char * speech_result_text(const speech_result * result);

/**
 * The number of segments of a recognition that set timestamps: runs of tokens that end where the model's file says a
 * sentence ends (a separator at the end of a word, as NeMo cuts, or a break after any token, for a language written
 * without spaces), or with the last token. 0 without timestamps and for a synthesis.
 */
SPEECH_API size_t speech_result_segment_count(const speech_result * result);

/**
 * The segment at `index`: the start of its first token and the end of its last in seconds from the start of the
 * audio, and its text. The texts of the segments, joined in order, are the result's text.
 */
SPEECH_API speech_status speech_result_segment(const speech_result * result, size_t index, double * start, double * end,
                                               const char ** text);

/** The number of tokens of a recognition that set timestamps; 0 without timestamps and for a synthesis. */
SPEECH_API size_t speech_result_token_count(const speech_result * result);

/**
 * The token at `index`: its start in seconds, the encoder frame the decoder emitted it on; its end, the frame that
 * its predicted duration reaches for a model that predicts one, or the next frame; and its text as it stands in the
 * result's text. A punctuation mark of a model that predicts durations starts and ends where the token before it
 * ends, as NeMo times it. The texts of the tokens, joined in order, are the result's text.
 */
SPEECH_API speech_status speech_result_token(const speech_result * result, size_t index, double * start, double * end,
                                             const char ** text);

#ifdef __cplusplus
}
#endif

#endif
