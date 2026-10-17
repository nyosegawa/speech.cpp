#pragma once

#include <string>
#include <vector>

#include "command-line.h"

// The subcommands of `speech`, each declared with its flags and the function that runs it.

Command tts_command();
Command asr_command();
Command vad_command();
Command voice_command();
Command info_command();
Command devices_command();
Command models_command();
Command pull_command();
Command rm_command();
Command quantize_command();
Command update_command();
Command serve_command();
Command worker_command();

/** The line that ends the loading of models: "Loaded Qwen3-ASR-0.6B on MTL0 and silero-vad on CPU in 0.24 s". */
std::string loaded_line(const std::vector<const speech_model_info *> & models, double seconds);
