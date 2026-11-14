// The checks of a dialogue model through the C API: loading without a warm-up, the information as the header declares it
// and as the file says, the options it takes and refuses, a voice from a WAVE file and the same voice made into a voice
// file, and conversations: the first frame from the start, a frame for each whole frame heard, the same frames bit for
// bit however the audio is cut into pieces and whichever form the voice takes, the audio resampled from another rate,
// max_seconds, the callback's stop at the first frame and later, and the refusals of the start and of what follows a
// conversation's end.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"

#define MAX_FRAMES 64

/** The frames a conversation gave: their audio end to end, their count and their text, and when to stop. */
typedef struct {
    float * samples;
    size_t n, cap;
    int frames, frame_samples, bad_length, outside, stop_at;
    char text[4096];
} Frames;

static int on_frame(const float * samples, size_t n, const char * text, void * user_data) {
    Frames * f = (Frames *) user_data;
    if (f->n + n > f->cap) {
        f->cap = (f->n + n) * 2;
        f->samples = (float *) realloc(f->samples, f->cap * sizeof(float));
    }
    memcpy(f->samples + f->n, samples, n * sizeof(float));
    f->n += n;
    if ((int) n != f->frame_samples) f->bad_length++;
    for (size_t i = 0; i < n; i++) f->outside += samples[i] < -1.0f || samples[i] > 1.0f;
    strncat(f->text, text, sizeof f->text - strlen(f->text) - 1);
    f->frames++;
    return f->stop_at > 0 && f->frames >= f->stop_at;
}

/** Writes mono float samples at `rate` as a WAVE file of 32-bit floats. */
static int write_wave(const char * path, const float * samples, size_t n, int rate) {
    FILE * f = open_utf8(path, "wb");
    if (!f) return 0;
    const uint32_t data = (uint32_t) (n * 4), riff = 36 + data, fmt = 16, byte_rate = (uint32_t) rate * 4, sample_rate = (uint32_t) rate;
    const uint16_t format = 3, channels = 1, align = 4, bits = 32;
    fwrite("RIFF", 1, 4, f);
    fwrite(&riff, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    fwrite(&fmt, 4, 1, f);
    fwrite(&format, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data, 4, 1, f);
    fwrite(samples, 4, n, f);
    return fclose(f) == 0;
}

static speech_request * conversation_request(speech_model * model, const char * voice, double max_seconds) {
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK) return NULL;
    speech_request_set_string(r, SPEECH_OPT_LANGUAGE, "ja");
    speech_request_set_string(r, SPEECH_OPT_NAME, "ハル");
    speech_request_set_string(r, SPEECH_OPT_PERSONA, "落ち着いた話し方をする大学生。");
    speech_request_set_string(r, SPEECH_OPT_BACKCHANNEL, "mid");
    if (voice) speech_request_set_string(r, SPEECH_OPT_VOICE, voice);
    speech_request_set_int(r, SPEECH_OPT_SEED, 7);
    speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, max_seconds);
    return r;
}

/**
 * A conversation of `request` that hears `audio` at `rate`, given in pieces of the lengths `pieces` repeats, ended
 * where `end` is set; returns the status of its last call and the frames in `f`.
 */
static speech_status converse(const speech_request * request, const float * audio, size_t n, int rate, const size_t * pieces, int n_pieces,
                              int end, Frames * f) {
    speech_dialogue * d = NULL;
    speech_status s = speech_dialogue_start(request, rate, on_frame, f, &d);
    for (size_t at = 0, i = 0; s == SPEECH_OK && at < n; i++) {
        const size_t piece = pieces[i % (size_t) n_pieces] < n - at ? pieces[i % (size_t) n_pieces] : n - at;
        s = speech_dialogue_push(d, audio + at, piece);
        at += piece;
    }
    if (s == SPEECH_OK && end) s = speech_dialogue_end(d);
    speech_dialogue_free(d);
    return s;
}

static int same_frames(const Frames * a, const Frames * b, const char * what) {
    if (a->frames != b->frames || a->n != b->n || memcmp(a->samples, b->samples, a->n * sizeof(float)) != 0 || strcmp(a->text, b->text) != 0) {
        fprintf(stderr, "FAIL: %s: %d and %d frames, which differ\n", what, a->frames, b->frames);
        return 0;
    }
    printf("%s: the same %d frames bit for bit\n", what, a->frames);
    return 1;
}

static int check_loaded(speech_model * model, const char * model_path, const char * dump, const char * scratch) {
    int bad = 0;
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const int rate = speech_model_info_sample_rate(info), frame = speech_model_info_frame_samples(info);
    if (speech_model_info_task(info) != SPEECH_TASK_DIALOGUE || rate != 24000 || frame != 3200) {
        fprintf(stderr, "FAIL: the model is not a dialogue model of 3200 samples a frame at 24 kHz\n");
        bad++;
    }
    static const speech_option options[] = {SPEECH_OPT_VOICE, SPEECH_OPT_LANGUAGE, SPEECH_OPT_SEED, SPEECH_OPT_MAX_SECONDS, SPEECH_OPT_TOP_K,
                                            SPEECH_OPT_TOP_P, SPEECH_OPT_TEMPERATURE, SPEECH_OPT_PERSONA, SPEECH_OPT_NAME, SPEECH_OPT_GENDER,
                                            SPEECH_OPT_BACKCHANNEL, SPEECH_OPT_OTHER_NAME, SPEECH_OPT_RELATIONSHIP, SPEECH_OPT_SITUATION,
                                            SPEECH_OPT_GUIDANCE_SCALE, SPEECH_OPT_EOP_BIAS, SPEECH_OPT_SPEAK_FIRST};
    const size_t count = speech_model_info_option_count(info);
    for (size_t i = 0; i < count || i < sizeof options / sizeof options[0]; i++) {
        if (i >= count || i >= sizeof options / sizeof options[0] || speech_model_info_option(info, i) != options[i]) {
            fprintf(stderr, "FAIL: the model's option %zu is not the one Haru declares\n", i);
            bad++;
            break;
        }
    }
    const char * json = speech_model_info_json(info);
    if (!strstr(json, "\"task\":\"dialogue\",\"sample_rate\":24000") || !strstr(json, "\"frame_samples\":3200") || !speech_model_info_voice_files(info) ||
        !speech_model_info_voice_recordings(info) || speech_model_info_voice_count(info) != 0) {
        fprintf(stderr, "FAIL: the information does not say a dialogue model of frames of 3200 samples that takes voice files and recordings\n");
        bad++;
    }
    speech_model_info_free(info);
    if (check_info_matches(model_path, model) != 0 || check_option_refusals(model) != 0) return 1;

    size_t n_ref = 0, n_user = 0;
    char path[4096], wav[4096], short_wav[4096], voice_file[4096];
    snprintf(path, sizeof path, "%s/reference_audio.npy", dump);
    float * reference = read_samples(path, &n_ref);
    snprintf(path, sizeof path, "%s/user_audio.npy", dump);
    float * user = read_samples(path, &n_user);
    if (!reference || !user) return fail("the dump's reference_audio.npy and user_audio.npy");
    snprintf(wav, sizeof wav, "%s/haru-reference.wav", scratch);
    snprintf(short_wav, sizeof short_wav, "%s/haru-short.wav", scratch);
    snprintf(voice_file, sizeof voice_file, "%s/haru-voice.gguf", scratch);
    if (!write_wave(wav, reference, n_ref, rate) || !write_wave(short_wav, reference, (size_t) rate, rate)) return fail("writing the WAVE files");

    // The voices: a WAVE file, the same made into a voice file, and a recording too short.
    bad += !expect(speech_voice_add(model, "ref", wav), SPEECH_OK, NULL, "a voice from a WAVE file");
    bad += !expect(speech_voice_make(model_path, wav, voice_file, NULL), SPEECH_OK, NULL, "a voice file from the WAVE file");
    bad += !expect(speech_voice_add(model, "file", voice_file), SPEECH_OK, NULL, "the voice file");
    bad += !expect(speech_voice_add(model, "short", short_wav), SPEECH_ERROR_OUT_OF_RANGE, "voice_path", "a recording of 1 s");

    // The refusals of a start and of a request's values.
    speech_request * r = conversation_request(model, "ref", 2.0);
    speech_dialogue * d = (speech_dialogue *) 1;
    bad += !expect(speech_dialogue_start(r, rate, NULL, NULL, &d), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a start without a callback");
    bad += d != NULL;
    bad += !expect(speech_dialogue_start(r, 0, on_frame, NULL, &d), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a start at 0 Hz");
    bad += !expect(speech_request_set_string(r, SPEECH_OPT_GENDER, "robot"), SPEECH_ERROR_OUT_OF_RANGE, "gender", "a gender the card does not offer");
    bad += !expect(speech_request_set_float(r, SPEECH_OPT_GUIDANCE_SCALE, 2.5), SPEECH_ERROR_OUT_OF_RANGE, "guidance_scale", "a guidance past 2");
    speech_request * card_without_language = conversation_request(model, NULL, 2.0);
    speech_request_set_string(card_without_language, SPEECH_OPT_LANGUAGE, "auto");
    bad += !expect(speech_dialogue_start(card_without_language, rate, on_frame, NULL, &d), SPEECH_ERROR_INVALID_ARGUMENT, "language",
                   "a card without a language");
    speech_request_free(card_without_language);
    speech_request * too_short = conversation_request(model, NULL, 0.1);
    bad += !expect(speech_dialogue_start(too_short, rate, on_frame, NULL, &d), SPEECH_ERROR_OUT_OF_RANGE, "max_seconds", "max_seconds below a frame");
    speech_request_free(too_short);
    speech_request * with_audio = conversation_request(model, NULL, 2.0);
    speech_request_set_audio(with_audio, user, 1000, rate);
    bad += !expect(speech_dialogue_start(with_audio, rate, on_frame, NULL, &d), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a request with audio");
    speech_request_free(with_audio);

    // A conversation a frame at a time, and the same in pieces of other lengths and with the voice file.
    const double seconds = 2.0;
    const int most = (int) (seconds * rate / frame);
    const size_t heard = (size_t) (most - 1) * (size_t) frame + (size_t) frame / 2;
    const size_t whole[] = {(size_t) frame}, odd[] = {1, 777, 5000, 3200, 31};
    Frames a = {0}, b = {0}, c = {0};
    a.frame_samples = b.frame_samples = c.frame_samples = frame;
    double t0 = now_seconds();
    bad += !expect(converse(r, user, heard, rate, whole, 1, 1, &a), SPEECH_OK, NULL, "a conversation a frame at a time");
    printf("%d frames in %.2f s\n", a.frames, now_seconds() - t0);
    if (a.frames != most || a.bad_length || a.outside) {
        fprintf(stderr, "FAIL: %d frames where %d were due, %d of the wrong length, %d samples outside [-1, 1]\n", a.frames, most, a.bad_length, a.outside);
        bad++;
    }
    bad += !expect(converse(r, user, heard, rate, odd, 5, 1, &b), SPEECH_OK, NULL, "the same in pieces of other lengths");
    bad += !same_frames(&a, &b, "pieces of other lengths");
    speech_request_set_string(r, SPEECH_OPT_VOICE, "file");
    bad += !expect(converse(r, user, heard, rate, whole, 1, 1, &c), SPEECH_OK, NULL, "the same with the voice file");
    bad += !same_frames(&a, &c, "the voice file");

    // More audio than max_seconds holds, the audio at another rate, and the callback's stops.
    Frames e = {0}, f = {0}, g = {0}, h = {0};
    e.frame_samples = f.frame_samples = g.frame_samples = h.frame_samples = frame;
    bad += !expect(converse(r, user, (size_t) (most + 3) * (size_t) frame, rate, whole, 1, 1, &e), SPEECH_ERROR_OUT_OF_RANGE, "max_seconds",
                   "more audio than max_seconds holds");
    bad += e.frames != most;
    float * twice = (float *) malloc(2 * heard * sizeof(float));
    for (size_t i = 0; i < heard; i++) twice[2 * i] = twice[2 * i + 1] = user[i];
    bad += !expect(converse(r, twice, 2 * heard, 2 * rate, odd, 5, 1, &f), SPEECH_OK, NULL, "the audio at twice the model's rate");
    if (f.frames != most || f.bad_length) {
        fprintf(stderr, "FAIL: %d frames from audio at %d Hz where %d were due\n", f.frames, 2 * rate, most);
        bad++;
    }
    free(twice);
    g.stop_at = 1;
    speech_dialogue * stopped = NULL;
    bad += !expect(speech_dialogue_start(r, rate, on_frame, &g, &stopped), SPEECH_CANCELLED, NULL, "a stop at the first frame");
    bad += !stopped || !expect(speech_dialogue_push(stopped, user, (size_t) frame), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a push after the stop");
    speech_dialogue_free(stopped);
    h.stop_at = 3;
    bad += !expect(converse(r, user, heard, rate, whole, 1, 1, &h), SPEECH_CANCELLED, NULL, "a stop at the third frame");
    bad += h.frames != 3;
    speech_dialogue * ended = NULL;
    Frames k = {0};
    k.frame_samples = frame;
    bad += !expect(speech_dialogue_start(r, rate, on_frame, &k, &ended), SPEECH_OK, NULL, "a conversation to end");
    bad += !expect(speech_dialogue_end(ended), SPEECH_OK, NULL, "its end");
    bad += !expect(speech_dialogue_push(ended, user, (size_t) frame), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a push after the end");
    speech_dialogue_free(ended);
    speech_request_free(r);

    printf("the text: %s\n", a.text);
    free(a.samples);
    free(b.samples);
    free(c.samples);
    free(e.samples);
    free(f.samples);
    free(g.samples);
    free(h.samples);
    free(k.samples);
    free(reference);
    free(user);
    return bad ? 1 : 0;
}

int check_dialogue(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "give a dump folder of reference/haru/dump.py and a scratch folder\n");
        return 2;
    }
    const char * model_path = argv[0];
    speech_load_params * params = NULL;
    if (speech_load_params_new(&params) != SPEECH_OK) return fail("speech_load_params_new");
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            if (speech_load_params_set_device(params, argv[++i]) != SPEECH_OK) return fail("speech_load_params_set_device");
        } else {
            fprintf(stderr, "unknown or incomplete option %s\n", argv[i]);
            return 2;
        }
    }
    speech_model * model = NULL;
    const double load_start = now_seconds();
    if (speech_model_load(model_path, params, &model) != SPEECH_OK) return fail("speech_model_load");
    speech_load_params_free(params);
    printf("loaded without a warm-up in %.2f s\n", now_seconds() - load_start);
    const int status = check_loaded(model, model_path, argv[1], argv[2]);
    speech_model_free(model);
    if (status == 0) printf("ok\n");
    return status;
}
