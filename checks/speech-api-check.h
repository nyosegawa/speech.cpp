#pragma once

#include <stddef.h>
#include <stdio.h>

#include "speech.h"

/** Opens a file whose path is UTF-8, as fopen() does with `mode`; on Windows fopen() reads the ANSI code page. */
FILE * open_utf8(const char * path, const char * mode);

/** Prints a failure with the last error's status, option and message, and returns 1. */
int fail(const char * what);

/**
 * Whether `got` is `want` and the last error names `option` (NULL for none); prints the outcome, the status's name and
 * the message, and returns 1 when it is as expected.
 */
int expect(speech_status got, speech_status want, const char * option, const char * what);

double now_seconds(void);
void sleep_seconds(double seconds);

/** A thread running `body(arg)`, started and joined in the platform's way. */
typedef struct {
    void * handle;
} Thread;
void thread_start(Thread * thread, void (*body)(void * arg), void * arg);
void thread_join(Thread * thread);

/** A mutex and a condition variable in the platform's way. */
typedef struct {
    void * lock;
    void * changed;
} Monitor;
void monitor_init(Monitor * m);
void monitor_free(Monitor * m);
void monitor_lock(Monitor * m);
void monitor_unlock(Monitor * m);
void monitor_wait(Monitor * m);
void monitor_signal(Monitor * m);

/**
 * Checks the versions, the names of the statuses, the stop reasons and the options, and the devices. Returns nonzero
 * on a failure.
 */
int check_library(void);

/** Checks the load parameters and the loads of `model_path` that are refused. Returns nonzero on a failure. */
int check_load_refusals(const char * model_path);

/**
 * Checks the information of the model file at `path` read without loading against the information of `model`, which
 * was loaded from it and has no voice added: the same JSON object but for the device and the threads, which only the
 * loaded model's has. Returns nonzero on a failure.
 */
int check_info_matches(const char * path, const speech_model * model);

/**
 * Checks that a request for `model` refuses each value of every option that the model does not take, or takes but not
 * at that value, with the category and the option's name, and accepts each option's neutral value. Returns nonzero on
 * a failure.
 */
int check_option_refusals(speech_model * model);

/**
 * The progress a request reports: how many reports, the first and the last, and whether one fell below the one before.
 * It stops the request once `stop_at` is reached when that is above 0.
 */
typedef struct {
    int n, fell;
    double first, last;
    double stop_at;
} Progress;
int record_progress(double done, void * user_data);

/** Whether the recorded progress rose from 0 or more to `last` without falling; prints a failure when not. */
int progress_rises(const Progress * p, double last, const char * what);

/** The checks of a recognition model: `argv` is <model.gguf> <dump folder>... [--device NAME]. Returns the exit status. */
int check_recognition(int argc, char ** argv);
