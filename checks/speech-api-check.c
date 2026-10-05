/*
 * Checks that the C API alone, through the shared libspeech, does what a program needs. It is written in C so that
 * speech.h is checked to be plain C.
 *
 * With a synthesis model: reports the release and API it was built as, lists the devices, optionally makes an
 * Irodori-TTS voice file and loads it, loads the model, describes it, speaks
 * one sentence into a WAVE file, takes or refuses the options of speed and length as the model can or cannot follow
 * them, stops a second request with speech_cancel() from another thread before it finishes, reports an unknown voice
 * as an error, and refuses to recognize speech.
 *
 * With a recognition model (transcribe): refuses to load it with voices or steps, loads it, describes it,
 * recognizes the audio of each dump of reference/fastconformer/dump.py and compares the text with the dump's text
 * byte for byte, refuses audio without a sample rate, no audio, a language the model does not recognize and a request
 * to speak, stops a request whose callback returns nonzero, and stops one with speech_cancel() from another thread
 * while the encoder runs.
 *
 * usage: speech-api-check <model.gguf> <out.wav> [--device NAME] [--voice NAME=FILE]...
 *                         [--make-voice <reference.wav> <voice.gguf>]
 *        speech-api-check transcribe <model.gguf> <dump folder>... [--device NAME]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"
#include "speech.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#else
#include <pthread.h>
#endif

#define MAX_VOICES 16

/** The audio of one request, grown as it arrives. */
typedef struct {
    float * samples;
    size_t n, capacity;
} Audio;

static int collect(const float * samples, size_t n, void * user_data) {
    Audio * a = (Audio *) user_data;
    if (a->n + n > a->capacity) {
        a->capacity = (a->n + n) * 2;
        a->samples = (float *) realloc(a->samples, a->capacity * sizeof(float));
        if (!a->samples) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
    }
    if (n) memcpy(a->samples + a->n, samples, n * sizeof(float));
    a->n += n;
    return 0;
}

#ifdef _WIN32
/** argv as UTF-8; the C runtime gives it in the ANSI code page, which cannot hold Japanese on most systems. */
static char ** utf8_argv(int * argc) {
    wchar_t ** wide = CommandLineToArgvW(GetCommandLineW(), argc);
    char ** argv = (char **) calloc((size_t) *argc + 1, sizeof(char *));
    for (int i = 0; i < *argc; i++) {
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, NULL, 0, NULL, NULL);
        argv[i] = (char *) malloc((size_t) bytes);
        WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, argv[i], bytes, NULL, NULL);
    }
    LocalFree(wide);
    return argv;
}

FILE * open_utf8(const char * path, const char * mode) {
    wchar_t wide[4096], wide_mode[8];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, 4096) || !MultiByteToWideChar(CP_UTF8, 0, mode, -1, wide_mode, 8)) return NULL;
    return _wfopen(wide, wide_mode);
}
#else
FILE * open_utf8(const char * path, const char * mode) {
    return fopen(path, mode);
}
#endif

static void put_u32(FILE * f, unsigned v) {
    const unsigned char b[4] = {(unsigned char) v, (unsigned char) (v >> 8), (unsigned char) (v >> 16), (unsigned char) (v >> 24)};
    fwrite(b, 1, 4, f);
}

static void put_u16(FILE * f, unsigned v) {
    const unsigned char b[2] = {(unsigned char) v, (unsigned char) (v >> 8)};
    fwrite(b, 1, 2, f);
}

/** Writes 16-bit mono PCM; returns 0 when the file cannot be written. */
static int write_wav(const char * path, const Audio * a, int rate) {
    FILE * f = open_utf8(path, "wb");
    if (!f) return 0;
    const unsigned bytes = (unsigned) (a->n * 2);
    fwrite("RIFF", 1, 4, f);
    put_u32(f, 36 + bytes);
    fwrite("WAVEfmt ", 1, 8, f);
    put_u32(f, 16);
    put_u16(f, 1);
    put_u16(f, 1);
    put_u32(f, (unsigned) rate);
    put_u32(f, (unsigned) rate * 2);
    put_u16(f, 2);
    put_u16(f, 16);
    fwrite("data", 1, 4, f);
    put_u32(f, bytes);
    for (size_t i = 0; i < a->n; i++) {
        float s = a->samples[i];
        s = s < -1.0f ? -1.0f : s > 1.0f ? 1.0f : s;
        const int v = (int) (s * 32767.0f + (s < 0 ? -0.5f : 0.5f));
        put_u16(f, (unsigned) (v & 0xFFFF));
    }
    return fclose(f) == 0;
}

/**
 * A request that another thread cancels once the first callback, with audio or without, has arrived; the
 * callback counts its calls after speech_cancel() returned.
 */
typedef struct {
    speech_model * model;
    int started;
    int after_cancel;
    int cancelled;
#ifdef _WIN32
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE changed;
#else
    pthread_mutex_t lock;
    pthread_cond_t changed;
#endif
} Cancelling;

static void lock(Cancelling * c) {
#ifdef _WIN32
    EnterCriticalSection(&c->lock);
#else
    pthread_mutex_lock(&c->lock);
#endif
}

static void unlock(Cancelling * c) {
#ifdef _WIN32
    LeaveCriticalSection(&c->lock);
#else
    pthread_mutex_unlock(&c->lock);
#endif
}

static void signal_changed(Cancelling * c) {
#ifdef _WIN32
    WakeAllConditionVariable(&c->changed);
#else
    pthread_cond_broadcast(&c->changed);
#endif
}

static void wait_changed(Cancelling * c) {
#ifdef _WIN32
    SleepConditionVariableCS(&c->changed, &c->lock, INFINITE);
#else
    pthread_cond_wait(&c->changed, &c->lock);
#endif
}

static int on_cancelling(const float * samples, size_t n, void * user_data) {
    Cancelling * c = (Cancelling *) user_data;
    (void) samples;
    (void) n;
    lock(c);
    c->started = 1;
    if (c->cancelled) c->after_cancel++;
    signal_changed(c);
    unlock(c);
    return 0;
}

#ifdef _WIN32
static DWORD WINAPI canceller(LPVOID user_data) {
#else
static void * canceller(void * user_data) {
#endif
    Cancelling * c = (Cancelling *) user_data;
    lock(c);
    while (!c->started) wait_changed(c);
    speech_cancel(c->model);
    c->cancelled = 1;
    unlock(c);
    return 0;
}

int fail(const char * what) {
    fprintf(stderr, "FAIL: %s: %s\n", what, speech_last_error());
    return 1;
}

/** Whether a request is refused with a message; prints the message. */
static int refused(speech_model * model, const speech_request * request, const char * what) {
    Audio none = {NULL, 0, 0};
    if (speech_synthesize(model, request, collect, &none) != SPEECH_ERROR || speech_last_error()[0] == '\0' || none.n != 0) {
        fprintf(stderr, "FAIL: %s is not refused with a message\n", what);
        free(none.samples);
        return 0;
    }
    printf("%s is refused: %s\n", what, speech_last_error());
    return 1;
}

/**
 * Irodori-TTS speaks a fixed length of 1 s in at most 1 s of audio and takes a speed, and refuses a speed out of
 * its range and seconds with a duration scale; Qwen3-TTS refuses a speed and a length. Returns nonzero on a failure.
 */
static int check_length_options(speech_model * model, const speech_request * sentence) {
    speech_request r = *sentence;
    if (!strcmp(speech_model_architecture(model), "irodori-tts")) {
        Audio audio = {NULL, 0, 0};
        r.seconds = 1;
        if (speech_synthesize(model, &r, collect, &audio) != SPEECH_OK) return fail("speech_synthesize with seconds");
        printf("a length of 1 s gave %.3f s of audio\n", (double) audio.n / speech_model_sample_rate(model));
        if (audio.n == 0 || audio.n > (size_t) speech_model_sample_rate(model)) {
            fprintf(stderr, "FAIL: a length of 1 s gave %zu samples\n", audio.n);
            return 1;
        }
        r.seconds = 0;
        r.speed = 1.5;
        audio.n = 0;
        if (speech_synthesize(model, &r, collect, &audio) != SPEECH_OK) return fail("speech_synthesize with speed");
        printf("speed 1.5 gave %.3f s of audio\n", (double) audio.n / speech_model_sample_rate(model));
        free(audio.samples);
        r.speed = 5;
        if (!refused(model, &r, "a speed of 5")) return 1;
        r.speed = 1;
        r.seconds = 2;
        r.duration_scale = 1.2;
        if (!refused(model, &r, "seconds with a duration scale")) return 1;
        r.duration_scale = 1;
        r.seconds = 60;
        return refused(model, &r, "a length of 60 s") ? 0 : 1;
    }
    r.speed = 1.5;
    if (!refused(model, &r, "a speed on Qwen3-TTS")) return 1;
    r.speed = 1;
    r.seconds = 2;
    if (!refused(model, &r, "a length on Qwen3-TTS")) return 1;
    r.seconds = 0;
    r.duration_scale = 0.8;
    return refused(model, &r, "a duration scale on Qwen3-TTS") ? 0 : 1;
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    argv = utf8_argv(&argc);
#endif
    const int recognition = argc > 1 && !strcmp(argv[1], "transcribe");
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <model.gguf> <out.wav> [--device NAME] [--voice NAME=FILE]... "
                "[--make-voice <reference.wav> <voice.gguf>]\n"
                "       %s transcribe <model.gguf> <dump folder>... [--device NAME]\n",
                argv[0], argv[0]);
        return 2;
    }
    if (speech_api_version() != SPEECH_API_VERSION) {
        fprintf(stderr, "FAIL: the library has API version %d, the header %d\n", speech_api_version(), SPEECH_API_VERSION);
        return 1;
    }
    if (strcmp(speech_version(), SPEECH_EXPECTED_VERSION) != 0) {
        fprintf(stderr, "FAIL: the library is speech.cpp %s, the check was built for %s\n", speech_version(), SPEECH_EXPECTED_VERSION);
        return 1;
    }
    printf("speech.cpp %s, API version %d\n", speech_version(), speech_api_version());
    if (recognition) return check_recognition(argc - 2, argv + 2);

    speech_model_params params = speech_model_default_params();
    params.model_path = argv[1];
    speech_voice_source voices[MAX_VOICES];
    size_t n_voices = 0;
    const char * reference = NULL, * made = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            params.device = argv[++i];
        } else if (!strcmp(argv[i], "--voice") && i + 1 < argc && n_voices < MAX_VOICES - 1) {
            char * eq = strchr(argv[++i], '=');
            if (!eq) {
                fprintf(stderr, "--voice takes NAME=FILE\n");
                return 2;
            }
            *eq = '\0';
            voices[n_voices].name = argv[i];
            voices[n_voices].path = eq + 1;
            n_voices++;
        } else if (!strcmp(argv[i], "--make-voice") && i + 2 < argc) {
            reference = argv[++i];
            made = argv[++i];
        } else {
            fprintf(stderr, "unknown or incomplete option %s\n", argv[i]);
            return 2;
        }
    }

    for (size_t i = 0; i < speech_device_count(); i++) {
        speech_device d;
        if (speech_device_get(i, &d) != SPEECH_OK) return fail("speech_device_get");
        printf("device %s (%s), kind %d, %llu of %llu bytes free\n", d.name, d.description, (int) d.kind,
               (unsigned long long) d.memory_free, (unsigned long long) d.memory_total);
    }
    {
        speech_device d;
        if (speech_device_get(speech_device_count(), &d) != SPEECH_ERROR) {
            fprintf(stderr, "FAIL: a device past the last one is not an error\n");
            return 1;
        }
    }

    if (made) {
        if (speech_make_voice(&params, reference, made) != SPEECH_OK) return fail("speech_make_voice");
        printf("made the voice file %s\n", made);
        voices[n_voices].name = "made";
        voices[n_voices].path = made;
        n_voices++;
    }
    params.voices = voices;
    params.n_voices = n_voices;

    speech_model * model = NULL;
    if (speech_model_load(&params, &model) != SPEECH_OK) return fail("speech_model_load");
    if (speech_model_task(model) != SPEECH_TASK_SYNTHESIS) {
        fprintf(stderr, "FAIL: a synthesis model reports the task %d\n", (int) speech_model_task(model));
        return 1;
    }
    printf("%s (%s) on %s: %d Hz, streaming by %s, steps %d, language selectable %d\n", speech_model_name(model),
           speech_model_architecture(model), speech_model_backend(model), speech_model_sample_rate(model),
           speech_model_streaming(model) == SPEECH_STREAMING_FRAME ? "frame" : "sentence", speech_model_steps(model),
           speech_model_language_selectable(model));
    for (size_t i = 0; i < speech_model_voice_count(model); i++) printf("  voice %s\n", speech_model_voice(model, i));
    for (size_t i = 0; i < speech_model_language_count(model); i++) printf("  language %s\n", speech_model_language(model, i));
    if (speech_model_voice_count(model) == 0 || speech_model_voice(model, speech_model_voice_count(model)) != NULL) {
        fprintf(stderr, "FAIL: the voices are not listed as declared\n");
        return 1;
    }

    const char * voice = made ? "made" : speech_model_voice(model, 0);
    speech_request request = speech_request_default();
    request.text = "明日の東京は晴れで、最高気温は二十四度の予報です。";
    request.voice = voice;
    request.language = "ja";
    request.seed = 7;
    Audio audio = {NULL, 0, 0};
    if (speech_synthesize(model, &request, collect, &audio) != SPEECH_OK) return fail("speech_synthesize");
    if (audio.n == 0) {
        fprintf(stderr, "FAIL: the sentence gave no audio\n");
        return 1;
    }
    if (!write_wav(argv[2], &audio, speech_model_sample_rate(model))) {
        fprintf(stderr, "FAIL: cannot write %s\n", argv[2]);
        return 1;
    }
    printf("spoke %.2f s of audio in the voice %s into %s\n", (double) audio.n / speech_model_sample_rate(model), voice, argv[2]);

    if (check_length_options(model, &request) != 0) return 1;

    Cancelling c;
    memset(&c, 0, sizeof c);
    c.model = model;
#ifdef _WIN32
    InitializeCriticalSection(&c.lock);
    InitializeConditionVariable(&c.changed);
    HANDLE thread = CreateThread(NULL, 0, canceller, &c, 0, NULL);
#else
    pthread_mutex_init(&c.lock, NULL);
    pthread_cond_init(&c.changed, NULL);
    pthread_t thread;
    pthread_create(&thread, NULL, canceller, &c);
#endif
    speech_request longer = speech_request_default();
    longer.text = "これは途中で止める長めの文です。止まったら、残りの音声は届きません。";
    longer.voice = voice;
    longer.seed = 8;
    const speech_status stopped = speech_synthesize(model, &longer, on_cancelling, &c);
#ifdef _WIN32
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
#else
    pthread_join(thread, NULL);
#endif
    if (stopped != SPEECH_STOPPED) {
        fprintf(stderr, "FAIL: a cancelled request returned %d, not SPEECH_STOPPED\n", (int) stopped);
        return 1;
    }
    printf("cancelled from another thread: %d call(s) of the callback after the cancel\n", c.after_cancel);
    if (c.after_cancel > 1) {
        fprintf(stderr, "FAIL: the callback ran more than once after speech_cancel() returned\n");
        return 1;
    }

    speech_request unknown = speech_request_default();
    unknown.text = "声の名前が違います。";
    unknown.voice = "no-such-voice";
    unknown.seed = 9;
    if (speech_synthesize(model, &unknown, collect, &audio) != SPEECH_ERROR || speech_last_error()[0] == '\0') {
        fprintf(stderr, "FAIL: an unknown voice is not an error with a message\n");
        return 1;
    }
    printf("an unknown voice is an error: %s\n", speech_last_error());

    {
        const float silence[1600] = {0};
        speech_transcription_request r = speech_transcription_request_default();
        r.samples = silence;
        r.n_samples = 1600;
        r.sample_rate = speech_model_sample_rate(model);
        if (speech_transcribe(model, &r, ignore_text, NULL) != SPEECH_ERROR || speech_last_error()[0] == '\0') {
            fprintf(stderr, "FAIL: speech_transcribe() on a synthesis model is not an error with a message\n");
            return 1;
        }
        printf("speech_transcribe() on a synthesis model is an error: %s\n", speech_last_error());
    }

    speech_model_free(model);
    free(audio.samples);
    printf("ok\n");
    return 0;
}
