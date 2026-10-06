/*
 * Checks that the C API alone, through the shared libspeech, does what a program needs. It is written in C so that
 * speech.h is checked to be plain C.
 *
 * In every mode: the versions, the names of the statuses, the stop reasons and the options, the devices, and the load
 * parameters and loads it refuses. With a synthesis model: optionally makes an Irodori-TTS voice file, loads the model
 * without a warm-up, compares the information read without loading with the loaded model's, adds the voices, refuses
 * each option's values with the category and the option's name and accepts the neutral ones, speaks one sentence into
 * a WAVE file, repeats a drawn seed's audio, checks the family's rules of the whole request (Qwen3-TTS's max_seconds and
 * longest text, Irodori-TTS's lengths, steps and progress), cancels a request from another thread and one before it
 * runs, and runs requests from two threads at once.
 *
 * With a recognition model (transcribe), speech-api-recognition.c.
 *
 * usage: speech-api-check <model.gguf> <out.wav> [--device NAME] [--voice NAME=FILE]...
 *                         [--make-voice <reference.wav> <voice.gguf>]
 *        speech-api-check transcribe <model.gguf> <dump folder>... [--device NAME]
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"
#include "speech.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

#define MAX_VOICES 16
#define SENTENCE "明日の東京は晴れで、最高気温は二十四度の予報です。"

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
    memcpy(a->samples + a->n, samples, n * sizeof(float));
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

static int log_counts[4];

/** Counts the library's and ggml's messages by level and prints the warnings and errors. */
static void on_log(speech_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if ((int) level >= 0 && (int) level < 4) log_counts[level]++;
    if (level >= SPEECH_LOG_WARN) fputs(text, stderr);
}

/** A request of `text` in `voice` (NULL for none) with `seed` (negative for none). */
static speech_request * new_request(speech_model * model, const char * text, const char * voice, int64_t seed) {
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK || (text && speech_request_set_text(r, text) != SPEECH_OK) ||
        (voice && speech_request_set_string(r, SPEECH_OPT_VOICE, voice) != SPEECH_OK) ||
        (seed >= 0 && speech_request_set_int(r, SPEECH_OPT_SEED, seed) != SPEECH_OK)) {
        fail("a request");
        exit(1);
    }
    return r;
}

/** Runs `r`, collecting its audio, and returns its status; the request is freed. */
static speech_status speak(speech_request * r, Audio * audio, speech_stop * stop, int64_t * seed) {
    const speech_status s = speech_synthesize(r, collect, audio);
    const speech_result * result = speech_request_result(r);
    if (stop) *stop = result ? speech_result_stop(result) : SPEECH_STOP_COMPLETE;
    if (seed) *seed = result ? speech_result_seed(result) : -1;
    if (result && speech_result_samples(result) != audio->n) {
        fprintf(stderr, "FAIL: a result counts %llu samples where the callback had %zu\n", (unsigned long long) speech_result_samples(result),
                audio->n);
        exit(1);
    }
    speech_request_free(r);
    return s;
}

/** Adds the voices of the command line, and checks the voices a model refuses. */
static int check_voices(speech_model * model, const char * model_path, char ** names, char ** paths, size_t n_voices, int irodori) {
    for (size_t i = 0; i < n_voices; i++) {
        const double start = now_seconds();
        if (speech_voice_add(model, names[i], paths[i]) != SPEECH_OK) return fail("speech_voice_add");
        printf("added the voice %s from %s in %.3f s\n", names[i], paths[i], now_seconds() - start);
    }
    int ok = 1;
    if (!irodori) {
        ok &= expect(speech_voice_add(model, "added", "voice.gguf"), SPEECH_ERROR_UNSUPPORTED, NULL, "a voice added to a model with its own");
        ok &= expect(speech_voice_make(model_path, "reference.wav", "voice.gguf", NULL), SPEECH_ERROR_UNSUPPORTED, "model_path",
                     "a voice file made for a model with its own voices");
        return ok ? 0 : 1;
    }
    ok &= expect(speech_voice_add(model, names[0], paths[0]), SPEECH_ERROR_INVALID_ARGUMENT, "name", "a voice under a name the model has");
    ok &= expect(speech_voice_add(model, "", paths[0]), SPEECH_ERROR_INVALID_ARGUMENT, "name", "a voice without a name");
    ok &= expect(speech_voice_add(model, "missing", "no-such-folder/voice.gguf"), SPEECH_ERROR_IO, "path", "a voice file that is not there");
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    ok &= speech_model_info_voice_count(info) == n_voices && !strcmp(speech_model_info_voice_name(info, n_voices - 1), names[n_voices - 1]) &&
          !strcmp(speech_model_info_voice_language(info, 0), "");
    if (!ok) fprintf(stderr, "FAIL: the information does not list the voices added\n");
    speech_model_info_free(info);
    return ok ? 0 : 1;
}

/** Qwen3-TTS: max_seconds stops the speech and a refused value leaves it, and a text past the longest is refused. */
static int check_qwen3_tts(speech_model * model, const speech_model_info * info, const char * voice) {
    const int rate = speech_model_info_sample_rate(info);
    speech_request * r = new_request(model, "これは最大の長さで止める、少し長めの文です。止まったところで終わります。", voice, 3);
    int ok = expect(speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, 0.5), SPEECH_OK, NULL, "max_seconds 0.5");
    ok &= expect(speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, 1e9), SPEECH_ERROR_OUT_OF_RANGE, "max_seconds", "max_seconds 1e9");
    Audio audio = {NULL, 0, 0};
    speech_stop stop;
    if (speak(r, &audio, &stop, NULL) != SPEECH_OK) return fail("a request with max_seconds");
    printf("max_seconds 0.5 gave %.3f s of audio, stopped at %s\n", (double) audio.n / rate, speech_stop_name(stop));
    ok &= stop == SPEECH_STOP_MAX_SECONDS && audio.n > 0 && audio.n <= (size_t) (0.5 * rate);
    if (!ok) fprintf(stderr, "FAIL: max_seconds does not stop the speech within it, or a refused value replaced it\n");
    free(audio.samples);

    size_t unit = 0, total = 0;
    if (speech_model_info_text_tokens(info, SENTENCE, &unit) != SPEECH_OK) return fail("speech_model_info_text_tokens");
    const size_t copies = speech_model_info_max_text_tokens(info) / unit + 2, length = strlen(SENTENCE);
    char * text = (char *) malloc(copies * length + 1);
    for (size_t i = 0; i < copies; i++) memcpy(text + i * length, SENTENCE, length);
    text[copies * length] = '\0';
    if (speech_model_info_text_tokens(info, text, &total) != SPEECH_OK) return fail("speech_model_info_text_tokens");
    printf("a text of %zu copies of the sentence is %zu tokens, the model takes %zu\n", copies, total, speech_model_info_max_text_tokens(info));
    Audio none = {NULL, 0, 0};
    ok &= total > speech_model_info_max_text_tokens(info) &&
          expect(speak(new_request(model, text, voice, 4), &none, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "text", "a text past the longest") &&
          none.n == 0;
    free(text);
    return ok ? 0 : 1;
}

/** Irodori-TTS: the length's rules, the steps, the progress of the sampler, and a text past the longest. */
static int check_irodori_tts(speech_model * model, const speech_model_info * info, const char * voice) {
    const int rate = speech_model_info_sample_rate(info);
    Audio audio = {NULL, 0, 0};
    speech_request * r = new_request(model, "三つ目です。", voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SECONDS, 1);
    if (speak(r, &audio, NULL, NULL) != SPEECH_OK) return fail("a request of 1 s");
    printf("seconds 1 gave %.3f s of audio\n", (double) audio.n / rate);
    int ok = audio.n > 0 && audio.n <= (size_t) rate;

    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SECONDS, 2);
    speech_request_set_float(r, SPEECH_OPT_DURATION_SCALE, 1.2);
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "seconds", "seconds with a duration scale");
    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SECONDS, 30);
    speech_request_set_float(r, SPEECH_OPT_SPEED, 0.5);
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "seconds", "30 s at a speed of 0.5");
    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_float(r, SPEECH_OPT_DURATION_SCALE, 50);
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "duration_scale", "a predicted length scaled by 50");
    r = new_request(model, "はい。", voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SPEED, 4);
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "speed", "a short predicted length at a speed of 4");

    Progress progress;
    memset(&progress, 0, sizeof progress);
    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_int(r, SPEECH_OPT_STEPS, 2);
    speech_request_set_progress(r, record_progress, &progress);
    audio.n = 0;
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_OK, NULL, "two steps") && progress.n == 3 && progress_rises(&progress, 1, "two steps");
    memset(&progress, 0, sizeof progress);
    progress.stop_at = 0.5;
    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_progress(r, record_progress, &progress);
    audio.n = 0;
    speech_stop stop;
    ok &= expect(speak(r, &audio, &stop, NULL), SPEECH_CANCELLED, NULL, "a progress callback that stops at 0.5") && audio.n == 0 &&
          stop == SPEECH_STOP_CANCELLED;
    free(audio.samples);

    size_t tokens = 0;
    char text[8192] = "";
    for (int i = 0; i < 120; i++) strcat(text, "あいうえお、");
    if (speech_model_info_text_tokens(info, text, &tokens) != SPEECH_OK) return fail("speech_model_info_text_tokens");
    printf("a text of %zu tokens, the model takes %zu\n", tokens, speech_model_info_max_text_tokens(info));
    Audio none = {NULL, 0, 0};
    ok &= tokens > speech_model_info_max_text_tokens(info) &&
          expect(speak(new_request(model, text, voice, 6), &none, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "text", "a text past the longest");
    if (!ok) fprintf(stderr, "FAIL: Irodori-TTS's rules of a request are not followed\n");
    return ok ? 0 : 1;
}

/** A request that another thread cancels once it has reported progress or passed audio. */
typedef struct {
    speech_request * request;
    Monitor monitor;
    int started, cancelled, after_cancel;
} Cancelling;

static void started(Cancelling * c) {
    monitor_lock(&c->monitor);
    c->started = 1;
    if (c->cancelled) c->after_cancel++;
    monitor_signal(&c->monitor);
    monitor_unlock(&c->monitor);
}

static int on_cancelling_audio(const float * samples, size_t n, void * user_data) {
    (void) samples;
    (void) n;
    started((Cancelling *) user_data);
    return 0;
}

static int on_cancelling_progress(double done, void * user_data) {
    (void) done;
    started((Cancelling *) user_data);
    return 0;
}

static void canceller(void * arg) {
    Cancelling * c = (Cancelling *) arg;
    monitor_lock(&c->monitor);
    while (!c->started) monitor_wait(&c->monitor);
    speech_request_cancel(c->request);
    c->cancelled = 1;
    monitor_unlock(&c->monitor);
}

/** A request run on a thread of its own, for two at once. */
typedef struct {
    speech_request * request;
    Audio audio;
    speech_status status;
} Concurrent;

static void run_concurrent(void * arg) {
    Concurrent * c = (Concurrent *) arg;
    c->status = speech_synthesize(c->request, collect, &c->audio);
}

/** Cancels a request from another thread and one before it runs, and runs two at once. */
static int check_threads(speech_model * model, const char * voice) {
    Cancelling c;
    memset(&c, 0, sizeof c);
    monitor_init(&c.monitor);
    c.request = new_request(model, "これは途中で止める長めの文です。止まったら、残りの音声は届きません。", voice, 8);
    speech_request_set_progress(c.request, on_cancelling_progress, &c);
    Thread thread;
    thread_start(&thread, canceller, &c);
    const speech_status status = speech_synthesize(c.request, on_cancelling_audio, &c);
    thread_join(&thread);
    const speech_result * result = speech_request_result(c.request);
    int ok = expect(status, SPEECH_CANCELLED, NULL, "a request cancelled from another thread") && result &&
             speech_result_stop(result) == SPEECH_STOP_CANCELLED && c.after_cancel <= 1;
    printf("cancelled from another thread: %llu samples, %d call(s) of a callback after the cancel\n",
           result ? (unsigned long long) speech_result_samples(result) : 0ULL, c.after_cancel);
    speech_request_free(c.request);
    monitor_free(&c.monitor);

    speech_request * early = new_request(model, SENTENCE, voice, 9);
    speech_request_cancel(early);
    Audio none = {NULL, 0, 0};
    speech_stop stop;
    ok &= expect(speak(early, &none, &stop, NULL), SPEECH_CANCELLED, NULL, "a request cancelled before it runs") && none.n == 0 &&
          stop == SPEECH_STOP_CANCELLED;

    Concurrent two[2];
    Thread threads[2];
    for (int i = 0; i < 2; i++) {
        memset(&two[i], 0, sizeof two[i]);
        two[i].request = new_request(model, i ? "二つ目です。" : "一つ目です。", voice, 10 + i);
        thread_start(&threads[i], run_concurrent, &two[i]);
    }
    for (int i = 0; i < 2; i++) {
        thread_join(&threads[i]);
        ok &= expect(two[i].status, SPEECH_OK, NULL, "one of two requests run at once") && two[i].audio.n > 0;
        speech_request_free(two[i].request);
        free(two[i].audio.samples);
    }
    if (ok) printf("two requests from two threads ran one after another\n");
    else fprintf(stderr, "FAIL: a request is not cancelled alone, or two requests at once do not both run\n");
    return ok ? 0 : 1;
}

/** The checks of a loaded synthesis model, which main frees whatever they find. */
static int check_model(speech_model * model, const char * model_path, const char * out_wav, char ** names, char ** paths, size_t n_voices,
                       int made) {
    if (check_info_matches(model_path, model) != 0) return 1;
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const int irodori = speech_model_info_voice_files(info);
    if (speech_model_info_task(info) != SPEECH_TASK_SYNTHESIS || speech_model_info_max_text_tokens(info) == 0 ||
        (irodori && (!speech_model_info_voice_codec(info) || n_voices == 0)) || (!irodori && speech_model_info_voice_count(info) == 0)) {
        fprintf(stderr, "FAIL: a synthesis model is not described as speech.h declares, or Irodori-TTS was given no voice\n");
        return 1;
    }
    if (check_voices(model, model_path, names, paths, n_voices, irodori) != 0 || check_option_refusals(model) != 0) return 1;
    speech_model_info_free(info);
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const int rate = speech_model_info_sample_rate(info);
    const char * voice = made ? "made" : speech_model_info_voice_name(info, 0);

    size_t tokens = 0;
    if (speech_model_info_text_tokens(info, SENTENCE, &tokens) != SPEECH_OK) return fail("speech_model_info_text_tokens");
    speech_request * r = new_request(model, SENTENCE, voice, 7);
    if (speech_request_set_string(r, SPEECH_OPT_LANGUAGE, "ja") != SPEECH_OK) return fail("the language ja");
    Audio audio = {NULL, 0, 0};
    speech_stop stop;
    int64_t seed = 0;
    const double start = now_seconds();
    if (speak(r, &audio, &stop, &seed) != SPEECH_OK) return fail("speech_synthesize");
    printf("spoke %zu tokens as %.2f s of audio in %.2f s in the voice %s, stopped at %s\n", tokens, (double) audio.n / rate,
           now_seconds() - start, voice, speech_stop_name(stop));
    if (audio.n == 0 || stop != SPEECH_STOP_COMPLETE || seed != 7 || !write_wav(out_wav, &audio, rate)) {
        fprintf(stderr, "FAIL: the sentence gave no audio, did not complete, did not report its seed, or %s cannot be written\n", out_wav);
        return 1;
    }

    Audio drawn = {NULL, 0, 0}, again = {NULL, 0, 0};
    if (speak(new_request(model, "二つ目の文です。", voice, -1), &drawn, NULL, &seed) != SPEECH_OK) return fail("a request without a seed");
    if (speak(new_request(model, "二つ目の文です。", voice, seed), &again, NULL, NULL) != SPEECH_OK) return fail("a request with the drawn seed");
    if (seed < 0 || seed > 9007199254740991LL || drawn.n != again.n || memcmp(drawn.samples, again.samples, drawn.n * sizeof(float)) != 0) {
        fprintf(stderr, "FAIL: the seed %lld that the library drew does not give the same audio again\n", (long long) seed);
        return 1;
    }
    printf("the drawn seed %lld gives the same %zu samples again\n", (long long) seed, again.n);

    r = new_request(model, "どの値も中立です。", voice, 12);
    int ok = expect(speech_request_set_float(r, SPEECH_OPT_SPEED, 1), SPEECH_OK, NULL, "speed 1") &&
             expect(speech_request_set_float(r, SPEECH_OPT_DURATION_SCALE, 1), SPEECH_OK, NULL, "duration_scale 1") &&
             expect(speech_request_set_string(r, SPEECH_OPT_LANGUAGE, "auto"), SPEECH_OK, NULL, "language auto") &&
             expect(speech_request_set_bool(r, SPEECH_OPT_TIMESTAMPS, 0), SPEECH_OK, NULL, "timestamps false");
    audio.n = 0;
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_OK, NULL, "a request of neutral values");
    r = new_request(model, NULL, voice, 13);
    ok &= expect(speech_request_set_audio(r, audio.samples, audio.n, rate), SPEECH_ERROR_UNSUPPORTED, "audio", "audio for a synthesis");
    ok &= expect(speech_transcribe(r), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_transcribe() of a synthesis model");
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "text", "a request without a text");
    r = new_request(model, SENTENCE, NULL, 14);
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "voice", "a request without a voice");
    r = new_request(model, "一回だけです。", voice, 15);
    audio.n = 0;
    ok &= expect(speech_synthesize(r, collect, &audio), SPEECH_OK, NULL, "a request's run") &&
          expect(speech_synthesize(r, collect, &audio), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "the same request run again");
    speech_request_free(r);
    if (!ok) return 1;
    free(audio.samples);
    free(drawn.samples);
    free(again.samples);

    if ((irodori ? check_irodori_tts(model, info, voice) : check_qwen3_tts(model, info, voice)) != 0) return 1;
    if (check_threads(model, voice) != 0) return 1;
    speech_model_info_free(info);
    return 0;
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
    speech_log_set(on_log, NULL);
    if (check_library() != 0 || check_load_refusals(argv[recognition ? 2 : 1]) != 0) return 1;
    if (recognition) return check_recognition(argc - 2, argv + 2);
    const char * model_path = argv[1];

    speech_load_params * params = NULL;
    if (speech_load_params_new(&params) != SPEECH_OK) return fail("speech_load_params_new");
    char * names[MAX_VOICES], * paths[MAX_VOICES];
    size_t n_voices = 0;
    const char * reference = NULL, * made = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            if (speech_load_params_set_device(params, argv[++i]) != SPEECH_OK) return fail("speech_load_params_set_device");
        } else if (!strcmp(argv[i], "--voice") && i + 1 < argc && n_voices < MAX_VOICES - 1) {
            char * eq = strchr(argv[++i], '=');
            if (!eq) {
                fprintf(stderr, "--voice takes NAME=FILE\n");
                return 2;
            }
            *eq = '\0';
            names[n_voices] = argv[i];
            paths[n_voices++] = eq + 1;
        } else if (!strcmp(argv[i], "--make-voice") && i + 2 < argc) {
            reference = argv[++i];
            made = argv[++i];
        } else {
            fprintf(stderr, "unknown or incomplete option %s\n", argv[i]);
            return 2;
        }
    }
    if (made) {
        const double start = now_seconds();
        if (speech_voice_make(model_path, reference, made, params) != SPEECH_OK) return fail("speech_voice_make");
        printf("made the voice file %s in %.3f s\n", made, now_seconds() - start);
        names[n_voices] = (char *) "made";
        paths[n_voices++] = (char *) made;
    }

    speech_model * model = NULL;
    double start = now_seconds();
    if (speech_model_load(model_path, params, &model) != SPEECH_OK) return fail("speech_model_load");
    speech_load_params_free(params);
    printf("loaded without a warm-up in %.2f s\n", now_seconds() - start);
    const int status = check_model(model, model_path, argv[2], names, paths, n_voices, made != NULL);
    speech_model_free(model);
    if (status == 0) {
        printf("log messages: %d debug, %d info, %d warnings, %d errors\n", log_counts[0], log_counts[1], log_counts[2], log_counts[3]);
        printf("ok\n");
    }
    return status;
}
