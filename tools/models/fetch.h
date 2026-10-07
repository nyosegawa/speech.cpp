#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

#include "catalog.h"

// Fetching a file of the catalog into the model folder with the system's curl, so that the system's proxies and
// certificates apply: curl writes the file's bytes to a pipe, which are appended to <file>.part and hashed as they
// come, and the file is renamed into place once its size and SHA-256 are the catalog's. A fetch that stops leaves the
// part, from whose end the next one resumes; two processes fetching the same file take turns on its lock. Only the
// command line and the server fetch: the library never reaches the network.

/** What a fetch reports while it runs: the bytes of the file there so far, and its size. It returns false to stop. */
using FetchProgress = std::function<bool(uint64_t done, uint64_t total)>;

/** What a fetch says besides its progress: that it waits for another process, or resumes a part. */
using FetchNote = std::function<void(const std::string & note)>;

/**
 * Fetches a file of the catalog unless it is in the folder, and returns its path. A failure throws an io Failure
 * that says what to do; a part whose bytes cannot be the file's is removed, and a fetch stopped by `progress` keeps it.
 */
std::filesystem::path fetch_model(const CatalogChoice & choice, const FetchProgress & progress, const FetchNote & note);

/** Fetches a file of the catalog as fetch_model() does, saying on stderr what it fetches, from where and how far it is. */
std::filesystem::path fetch_on_stderr(const CatalogChoice & choice);

/**
 * The model file a subcommand's model argument names, as a UTF-8 path: a path to a GGUF file as given, or the catalog's
 * file, fetched first when it is not in the folder.
 */
std::string model_file(const std::string & argument);
