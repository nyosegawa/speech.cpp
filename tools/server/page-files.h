#pragma once

#include <cstddef>

/** A file of the page as the build copies it into speech: its name in tools/server/page/ and its bytes. */
struct PageFile {
    const char * name;
    const unsigned char * bytes;
    size_t size;
};

/** The files of tools/server/page/, written into the build by CMake. */
extern const PageFile speech_page_files[];
extern const size_t speech_page_file_count;
