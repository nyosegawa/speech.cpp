#pragma once

#include <stdio.h>

/** Opens a file whose path is UTF-8, as fopen() does with `mode`; on Windows fopen() reads the ANSI code page. */
FILE * open_utf8(const char * path, const char * mode);

/** Prints a failure with speech_last_error() and returns 1. */
int fail(const char * what);

/** Ignores a request's text. */
int ignore_text(const char * text, void * user_data);

/**
 * The checks of a recognition model: `argv` is <model.gguf> <dump folder>... [--device NAME]. Returns the exit
 * status.
 */
int check_recognition(int argc, char ** argv);
