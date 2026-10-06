/*
 * The recognition half of speech-api-check: the C API with a recognition model, through the shared libspeech. Loads the
 * model without a warm-up, compares the information read without loading with the loaded model's, refuses each
 * option's values with the category and the option's name, recognizes the audio of each dump of
 * reference/fastconformer/dump.py and compares the text with the dump's text byte for byte, gives each its tokens and
 * segments with their times, recognizes audio at three times the model's rate, refuses what a recognition cannot take,
 * reports and stops on progress, and cancels a request from another thread while the encoder runs.
 */

#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"
#include "speech.h"

#define MAX_DUMPS 16

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

/**
 * Recognizes `n` samples at `rate` with `timestamps`, returning the status and the result's text in `text` (which the
 * caller frees), checking the tokens and segments against the text when they are asked for.
 */
static speech_status recognize(speech_model * model, const float * samples, size_t n, int rate, int timestamps, Progress * progress,
                               char ** text) {
    speech_request * r = NULL;
    *text = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, samples, n, rate) != SPEECH_OK ||
        speech_request_set_bool(r, SPEECH_OPT_TIMESTAMPS, timestamps) != SPEECH_OK ||
        (progress && speech_request_set_progress(r, record_progress, progress) != SPEECH_OK)) {
        speech_request_free(r);
        return SPEECH_ERROR_INTERNAL;
    }
    const speech_status status = speech_transcribe(r);
    const speech_result * result = speech_request_result(r);
    if (result) {
        const char * t = speech_result_text(result);
        *text = (char *) malloc(strlen(t) + 1);
        strcpy(*text, t);
    }
    if (status == SPEECH_OK && timestamps) {
        const size_t length = strlen(*text);
        char * joined_tokens = (char *) calloc(length + 1, 1), * joined_segments = (char *) calloc(length + 1, 1);
        const double seconds = (double) n / rate;
        double last = 0;
        int ok = 1;
        for (size_t i = 0; i < speech_result_token_count(result); i++) {
            double start = 0, end = 0;
            const char * piece = NULL;
            ok = ok && speech_result_token(result, i, &start, &end, &piece) == SPEECH_OK && start >= last - 1e-9 && end >= start &&
                 end <= seconds + 0.5 && strlen(joined_tokens) + strlen(piece) <= length;
            if (!ok) break;
            strcat(joined_tokens, piece);
            last = start;
        }
        printf("  %zu tokens, %zu segments:\n", speech_result_token_count(result), speech_result_segment_count(result));
        for (size_t i = 0; ok && i < speech_result_segment_count(result); i++) {
            double start = 0, end = 0;
            const char * piece = NULL;
            ok = speech_result_segment(result, i, &start, &end, &piece) == SPEECH_OK && end >= start && strlen(joined_segments) + strlen(piece) <= length;
            if (ok) {
                strcat(joined_segments, piece);
                printf("  %8.2f %8.2f %s\n", start, end, piece);
            }
        }
        ok = ok && !strcmp(joined_tokens, *text) && !strcmp(joined_segments, *text) && (length == 0 || speech_result_segment_count(result) > 0) &&
             speech_result_segment(result, speech_result_segment_count(result), NULL, NULL, NULL) == SPEECH_ERROR_INVALID_ARGUMENT;
        if (!ok) fprintf(stderr, "FAIL: the tokens and segments do not make the text, or their times do not rise within the audio\n");
        free(joined_tokens);
        free(joined_segments);
        if (!ok) {
            speech_request_free(r);
            return SPEECH_ERROR_INTERNAL;
        }
    }
    speech_request_free(r);
    return status;
}

static int ignore_audio(const float * samples, size_t n, void * user_data) {
    (void) samples;
    (void) n;
    (void) user_data;
    return 0;
}

/** A thread that cancels a request a fixed time after it starts, and notes when it did. */
typedef struct {
    speech_request * request;
    double delay, cancelled_at;
} Canceller;

static void cancel_later(void * arg) {
    Canceller * c = (Canceller *) arg;
    const double start = now_seconds();
    sleep_seconds(c->delay);
    speech_request_cancel(c->request);
    c->cancelled_at = now_seconds() - start;
}

/** What a recognition model refuses: options it does not take, the wrong input, and voices. */
static int check_refusals(speech_model * model, const char * model_path, const float * samples, size_t n, int rate) {
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK) return fail("speech_request_new");
    int ok = expect(speech_request_set_audio(r, samples, n, 0), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "audio without a sample rate");
    ok &= expect(speech_request_set_audio(r, samples, n, 44101), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "audio at 44101 Hz");
    ok &= expect(speech_request_set_audio(r, samples, 0, rate), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "no audio");
    ok &= expect(speech_request_set_audio(r, NULL, n, rate), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "NULL samples");
    ok &= expect(speech_request_set_text(r, "明日の東京は晴れです。"), SPEECH_ERROR_UNSUPPORTED, "text", "a text for a recognition");
    ok &= expect(speech_synthesize(r, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "speech_synthesize() without a callback");
    ok &= expect(speech_synthesize(r, ignore_audio, NULL), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_synthesize() of a recognition model");
    ok &= expect(speech_transcribe(r), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a recognition without audio");
    speech_request_free(r);
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, samples, 100, rate) != SPEECH_OK) return fail("a request");
    ok &= expect(speech_transcribe(r), SPEECH_ERROR_OUT_OF_RANGE, "audio", "100 samples of audio");
    speech_request_free(r);
    // A refused value leaves the one set before it.
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, samples, n, rate) != SPEECH_OK ||
        speech_request_set_bool(r, SPEECH_OPT_TIMESTAMPS, 1) != SPEECH_OK) {
        return fail("a request");
    }
    ok &= expect(speech_request_set_float(r, SPEECH_OPT_TIMESTAMPS, 0), SPEECH_ERROR_INVALID_ARGUMENT, "timestamps", "timestamps as a number");
    ok &= expect(speech_transcribe(r), SPEECH_OK, NULL, "a recognition after a refused value") &&
          speech_result_segment_count(speech_request_result(r)) > 0;
    speech_request_free(r);
    speech_model_info * info = NULL;
    size_t tokens = 0;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    ok &= expect(speech_model_info_text_tokens(info, "明日", &tokens), SPEECH_ERROR_UNSUPPORTED, NULL, "the tokens of a text for a recognition");
    speech_model_info_free(info);
    ok &= expect(speech_voice_add(model, "voice", "voice.gguf"), SPEECH_ERROR_UNSUPPORTED, NULL, "a voice for a recognition");
    ok &= expect(speech_voice_make(model_path, "reference.wav", "voice.gguf", NULL), SPEECH_ERROR_UNSUPPORTED, "model_path",
                 "a voice file for a recognition model");
    return ok ? 0 : 1;
}

/** The checks of a loaded recognition model, which check_recognition() frees whatever they find. */
static int check_loaded(speech_model * model, const char * model_path, const char ** dumps, int n_dumps) {
    if (check_info_matches(model_path, model) != 0 || check_option_refusals(model) != 0) return 1;
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const int rate = speech_model_info_sample_rate(info);
    if (speech_model_info_task(info) != SPEECH_TASK_RECOGNITION || speech_model_info_incremental(info) || speech_model_info_voice_count(info) ||
        speech_model_info_voice_files(info) || speech_model_info_max_text_tokens(info) || rate <= 0 || speech_model_info_language_count(info) == 0 ||
        speech_model_info_option_steers(info, SPEECH_OPT_LANGUAGE)) {
        fprintf(stderr, "FAIL: the recognition model is not described as speech.h declares\n");
        return 1;
    }
    speech_model_info_free(info);

    float * audio[MAX_DUMPS];
    size_t lengths[MAX_DUMPS];
    char * texts[MAX_DUMPS];
    int shortest = 0, longest = 0, ok = 1;
    for (int d = 0; d < n_dumps; d++) {
        size_t n = 0, text_size = 0;
        char path[4096];
        snprintf(path, sizeof path, "%s/text.txt", dumps[d]);
        audio[d] = read_audio(dumps[d], &n);
        lengths[d] = n;
        texts[d] = read_whole(path, &text_size);
        if (!audio[d] || !texts[d]) {
            fprintf(stderr, "FAIL: %s has no audio.npy of float32 samples or no text.txt\n", dumps[d]);
            return 1;
        }
        char * text = NULL;
        Progress progress;
        memset(&progress, 0, sizeof progress);
        const double start = now_seconds();
        if (recognize(model, audio[d], n, rate, 1, &progress, &text) != SPEECH_OK) return fail("speech_transcribe");
        const int equal = !strcmp(text, texts[d]);
        printf("%s: %.2f s of audio in %.3f s, text %s\n", dumps[d], (double) n / rate, now_seconds() - start, equal ? "equal to the dump's" : "DIFFERS");
        if (!equal) printf("  got  %s\n  want %s\n", text, texts[d]);
        ok = ok && equal && progress_rises(&progress, 1, "the recognition");
        free(text);
        if (n < lengths[shortest]) shortest = d;
        if (n > lengths[longest]) longest = d;
    }
    if (!ok) {
        fprintf(stderr, "FAIL: a text differs from the dump's, or the progress does not rise\n");
        return 1;
    }

    // Each sample three times is audio at three times the rate, which the library resamples to the model's.
    const size_t n = lengths[shortest];
    float * tripled = (float *) malloc(3 * n * sizeof(float));
    for (size_t i = 0; i < 3 * n; i++) tripled[i] = audio[shortest][i / 3];
    char * text = NULL;
    if (recognize(model, tripled, 3 * n, 3 * rate, 0, NULL, &text) != SPEECH_OK || !text[0]) return fail("audio at three times the rate");
    printf("the same audio at %d Hz: %s\n", 3 * rate, text);
    free(text);
    free(tripled);
    if (check_refusals(model, model_path, audio[shortest], n, rate) != 0) return 1;

    Progress stopping;
    memset(&stopping, 0, sizeof stopping);
    stopping.stop_at = 0.5;
    if (!expect(recognize(model, audio[shortest], n, rate, 1, &stopping, &text), SPEECH_CANCELLED, NULL, "a progress callback that stops at 0.5") ||
        strcmp(text, "") != 0) {
        fprintf(stderr, "FAIL: a stopped recognition has a text\n");
        return 1;
    }
    free(text);

    // The cancel comes 20 ms into the longest audio, within the encoder; the text comes within milliseconds of the
    // encoder's end, so a request still running well after the cancel must stop.
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, audio[longest], lengths[longest], rate) != SPEECH_OK) {
        return fail("a request");
    }
    Canceller c = {r, 0.02, 0};
    Thread thread;
    thread_start(&thread, cancel_later, &c);
    const double start = now_seconds();
    const speech_status status = speech_transcribe(r);
    const double took = now_seconds() - start;
    thread_join(&thread);
    const speech_result * result = speech_request_result(r);
    if (status == SPEECH_CANCELLED && result && !strcmp(speech_result_text(result), "") && speech_result_stop(result) == SPEECH_STOP_CANCELLED) {
        printf("cancelled from another thread after %.3f s: stopped at %.3f s without text\n", c.cancelled_at, took);
    } else if (status == SPEECH_OK && took < c.cancelled_at + 0.01) {
        printf("the request finished in %.3f s, before the cancel at %.3f s took effect; cancellation was not exercised\n", took, c.cancelled_at);
    } else {
        fprintf(stderr, "FAIL: a cancel at %.3f s did not stop a request that ran %.3f s (status %d)\n", c.cancelled_at, took, (int) status);
        return 1;
    }
    speech_request_free(r);

    if (recognize(model, audio[shortest], n, rate, 0, NULL, &text) != SPEECH_OK || strcmp(text, texts[shortest]) != 0) {
        fprintf(stderr, "FAIL: the request after a cancelled one does not give its text\n");
        return 1;
    }
    printf("the request after the cancelled one gives its text\n");
    free(text);
    for (int d = 0; d < n_dumps; d++) {
        free(audio[d]);
        free(texts[d]);
    }
    return 0;
}

int check_recognition(int argc, char ** argv) {
    const char * model_path = argv[0];
    speech_load_params * params = NULL;
    if (speech_load_params_new(&params) != SPEECH_OK) return fail("speech_load_params_new");
    const char * dumps[MAX_DUMPS];
    int n_dumps = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            if (speech_load_params_set_device(params, argv[++i]) != SPEECH_OK) return fail("speech_load_params_set_device");
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

    speech_model * model = NULL;
    const double load_start = now_seconds();
    if (speech_model_load(model_path, params, &model) != SPEECH_OK) return fail("speech_model_load");
    speech_load_params_free(params);
    printf("loaded without a warm-up in %.2f s\n", now_seconds() - load_start);
    const int status = check_loaded(model, model_path, dumps, n_dumps);
    speech_model_free(model);
    if (status == 0) printf("ok\n");
    return status;
}
