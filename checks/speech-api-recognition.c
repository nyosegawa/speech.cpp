/*
 * The recognition half of speech-api-check: the C API with a recognition model, through the shared libspeech.
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
#else
#include <pthread.h>
#include <time.h>
#endif

#define MAX_DUMPS 16
#define MAX_TEXT 65536

static double now_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER count, frequency;
    QueryPerformanceCounter(&count);
    QueryPerformanceFrequency(&frequency);
    return (double) count.QuadPart / (double) frequency.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
#endif
}

static void sleep_seconds(double seconds) {
#ifdef _WIN32
    Sleep((DWORD) (seconds * 1000));
#else
    struct timespec t;
    t.tv_sec = (time_t) seconds;
    t.tv_nsec = (long) ((seconds - (double) t.tv_sec) * 1e9);
    nanosleep(&t, NULL);
#endif
}

/** A whole file, NUL-terminated, or NULL when it cannot be read; `size` is its length. */
static char * read_whole(const char * path, size_t * size) {
    FILE * f = open_utf8(path, "rb");
    if (!f) return NULL;
    size_t capacity = 1 << 16, n = 0;
    char * data = (char *) malloc(capacity);
    for (;;) {
        n += fread(data + n, 1, capacity - n - 1, f);
        if (n < capacity - 1) break;
        capacity *= 2;
        data = (char *) realloc(data, capacity);
    }
    fclose(f);
    data[n] = '\0';
    *size = n;
    return data;
}

/**
 * The samples of a dump's audio.npy, a one-dimensional little-endian float32 array in NumPy's format 1.0, or NULL
 * when the file is not one.
 */
static float * read_audio(const char * dump, size_t * n) {
    char path[4096];
    snprintf(path, sizeof path, "%s/audio.npy", dump);
    size_t size = 0;
    char * data = read_whole(path, &size);
    if (!data) return NULL;
    const size_t header = size >= 10 ? (size_t) (unsigned char) data[8] | (size_t) (unsigned char) data[9] << 8 : 0;
    if (size < 10 || memcmp(data, "\x93NUMPY\x01", 7) != 0 || 10 + header > size || !strstr(data + 10, "'<f4'") ||
        !strstr(data + 10, "'fortran_order': False") || !strstr(data + 10, ",)")) {
        free(data);
        return NULL;
    }
    *n = (size - 10 - header) / sizeof(float);
    float * samples = (float *) malloc(*n * sizeof(float));
    memcpy(samples, data + 10 + header, *n * sizeof(float));
    free(data);
    return samples;
}

typedef struct {
    char text[MAX_TEXT];
    int calls;
    int stop;
} Text;

static int take_text(const char * text, void * user_data) {
    Text * t = (Text *) user_data;
    snprintf(t->text + strlen(t->text), MAX_TEXT - strlen(t->text), "%s", text);
    t->calls++;
    return t->stop;
}

static int ignore_audio(const float * samples, size_t n, void * user_data) {
    (void) samples;
    (void) n;
    (void) user_data;
    return 0;
}

int ignore_text(const char * text, void * user_data) {
    (void) text;
    (void) user_data;
    return 0;
}

/** Whether loading with `params` is an error whose message names `field`; prints the message. */
static int load_refused(const speech_model_params * params, const char * field, const char * what) {
    speech_model * model = NULL;
    if (speech_model_load(params, &model) != SPEECH_ERROR || model || !strstr(speech_last_error(), field)) {
        fprintf(stderr, "FAIL: %s is not an error that names %s\n", what, field);
        speech_model_free(model);
        return 0;
    }
    printf("%s is refused: %s\n", what, speech_last_error());
    return 1;
}

/** Whether a request is refused with a message and no text; prints the message. */
static int request_refused(speech_model * model, const speech_transcription_request * r, const char * what) {
    Text t = {{0}, 0, 0};
    if (speech_transcribe(model, r, take_text, &t) != SPEECH_ERROR || speech_last_error()[0] == '\0' || t.calls != 0) {
        fprintf(stderr, "FAIL: %s is not refused with a message\n", what);
        return 0;
    }
    printf("%s is refused: %s\n", what, speech_last_error());
    return 1;
}

/** A thread that cancels the model's request a fixed time after it starts, and notes when it did. */
typedef struct {
    speech_model * model;
    double delay, cancelled_at;
} Canceller;

#ifdef _WIN32
static DWORD WINAPI cancel_later(LPVOID user_data) {
#else
static void * cancel_later(void * user_data) {
#endif
    Canceller * c = (Canceller *) user_data;
    const double start = now_seconds();
    sleep_seconds(c->delay);
    speech_cancel(c->model);
    c->cancelled_at = now_seconds() - start;
    return 0;
}

int check_recognition(int argc, char ** argv) {
    speech_model_params params = speech_model_default_params();
    params.model_path = argv[0];
    const char * dumps[MAX_DUMPS];
    int n_dumps = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            params.device = argv[++i];
        } else if (argv[i][0] != '-' && n_dumps < MAX_DUMPS) {
            dumps[n_dumps++] = argv[i];
        } else {
            fprintf(stderr, "unknown or incomplete option %s\n", argv[i]);
            return 2;
        }
    }
    if (n_dumps == 0) {
        fprintf(stderr, "give at least one dump folder of reference/fastconformer/dump.py\n");
        return 2;
    }

    speech_voice_source voice = {"voice", "voice.gguf"};
    speech_model_params wrong = params;
    wrong.voices = &voice;
    wrong.n_voices = 1;
    if (!load_refused(&wrong, "voices", "a recognition model with a voice")) return 1;
    wrong = params;
    wrong.steps = 4;
    if (!load_refused(&wrong, "steps", "a recognition model with steps")) return 1;

    speech_model * model = NULL;
    const double load_start = now_seconds();
    if (speech_model_load(&params, &model) != SPEECH_OK) return fail("speech_model_load");
    const int rate = speech_model_sample_rate(model);
    printf("%s (%s) on %s in %.2f s: %d Hz, language selectable %d\n", speech_model_name(model), speech_model_architecture(model),
           speech_model_backend(model), now_seconds() - load_start, rate, speech_model_language_selectable(model));
    for (size_t i = 0; i < speech_model_language_count(model); i++) printf("  language %s\n", speech_model_language(model, i));
    if (speech_model_task(model) != SPEECH_TASK_RECOGNITION || speech_model_streaming(model) != SPEECH_STREAMING_NONE ||
        speech_model_voice_count(model) != 0 || speech_model_voice(model, 0) != NULL || speech_model_steps(model) != 0 ||
        rate <= 0 || speech_model_language_count(model) == 0) {
        fprintf(stderr, "FAIL: the recognition model is not described as speech.h declares\n");
        return 1;
    }

    float * audio[MAX_DUMPS];
    size_t lengths[MAX_DUMPS];
    char * texts[MAX_DUMPS];
    int shortest = 0, longest = 0, ok = 1;
    for (int d = 0; d < n_dumps; d++) {
        size_t n = 0, text_size = 0;
        float * samples = read_audio(dumps[d], &n);
        char path[4096];
        snprintf(path, sizeof path, "%s/text.txt", dumps[d]);
        char * want = read_whole(path, &text_size);
        audio[d] = samples;
        lengths[d] = n;
        texts[d] = want;
        if (!samples || !want) {
            fprintf(stderr, "FAIL: %s has no audio.npy of float32 samples or no text.txt\n", dumps[d]);
            return 1;
        }
        speech_transcription_request r = speech_transcription_request_default();
        r.samples = samples;
        r.n_samples = n;
        r.sample_rate = rate;
        r.language = speech_model_language(model, 0);
        Text t = {{0}, 0, 0};
        const double start = now_seconds();
        if (speech_transcribe(model, &r, take_text, &t) != SPEECH_OK) return fail("speech_transcribe");
        const int equal = t.calls == 1 && !strcmp(t.text, want);
        printf("%s: %.2f s of audio in %.3f s, text %s\n", dumps[d], (double) n / rate, now_seconds() - start,
               equal ? "equal to the dump's text" : "DIFFERS");
        if (!equal) printf("  got  %s\n  want %s\n", t.text, want);
        ok = ok && equal;
        if (n < lengths[shortest]) shortest = d;
        if (n > lengths[longest]) longest = d;
    }
    if (!ok) {
        fprintf(stderr, "FAIL: a text differs from the dump's\n");
        return 1;
    }

    speech_transcription_request r = speech_transcription_request_default();
    r.samples = audio[shortest];
    r.n_samples = lengths[shortest];
    r.sample_rate = 0;
    if (!request_refused(model, &r, "audio without a sample rate")) return 1;
    r.sample_rate = rate;
    r.n_samples = 0;
    if (!request_refused(model, &r, "no audio")) return 1;
    r.n_samples = lengths[shortest];
    r.language = "zz";
    if (!request_refused(model, &r, "a language the model does not recognize")) return 1;
    r.language = "auto";
    {
        speech_request speak = speech_request_default();
        speak.text = "明日の東京は晴れです。";
        speak.voice = "voice";
        if (speech_synthesize(model, &speak, ignore_audio, NULL) != SPEECH_ERROR || !strstr(speech_last_error(), "recognition")) {
            fprintf(stderr, "FAIL: speech_synthesize() on a recognition model is not an error that names its task\n");
            return 1;
        }
        printf("speech_synthesize() on a recognition model is an error: %s\n", speech_last_error());
    }

    Text stopping = {{0}, 0, 1};
    if (speech_transcribe(model, &r, take_text, &stopping) != SPEECH_STOPPED || stopping.calls != 1) {
        fprintf(stderr, "FAIL: a callback that returns nonzero does not stop the request\n");
        return 1;
    }
    printf("a callback that returns nonzero stops the request\n");

    // The cancel comes 20 ms into the longest audio, within the encoder; the text is passed within milliseconds of the
    // encoder's end, so a request still running well after the cancel must stop.
    Canceller c = {model, 0.02, 0};
    Text t = {{0}, 0, 0};
    r.samples = audio[longest];
    r.n_samples = lengths[longest];
#ifdef _WIN32
    HANDLE thread = CreateThread(NULL, 0, cancel_later, &c, 0, NULL);
#else
    pthread_t thread;
    pthread_create(&thread, NULL, cancel_later, &c);
#endif
    const double start = now_seconds();
    const speech_status status = speech_transcribe(model, &r, take_text, &t);
    const double took = now_seconds() - start;
#ifdef _WIN32
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
#else
    pthread_join(thread, NULL);
#endif
    if (status == SPEECH_STOPPED && t.calls == 0) {
        printf("cancelled from another thread after %.3f s: stopped at %.3f s without text\n", c.cancelled_at, took);
    } else if (status == SPEECH_OK && took < c.cancelled_at + 0.01) {
        printf("the request finished in %.3f s, before the cancel at %.3f s took effect; cancellation was not exercised\n", took,
               c.cancelled_at);
    } else {
        fprintf(stderr, "FAIL: a cancel at %.3f s did not stop a request that ran %.3f s (status %d, %d texts)\n", c.cancelled_at, took,
                (int) status, t.calls);
        return 1;
    }

    Text after = {{0}, 0, 0};
    r.samples = audio[shortest];
    r.n_samples = lengths[shortest];
    if (speech_transcribe(model, &r, take_text, &after) != SPEECH_OK || strcmp(after.text, texts[shortest]) != 0) {
        fprintf(stderr, "FAIL: the request after a cancelled one does not give its text\n");
        return 1;
    }
    printf("the request after the cancelled one gives its text\n");

    speech_model_free(model);
    for (int d = 0; d < n_dumps; d++) {
        free(audio[d]);
        free(texts[d]);
    }
    printf("ok\n");
    return 0;
}
