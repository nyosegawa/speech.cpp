"""The cut of transcription by regions, as tools/common/regions.cpp's RegionCut makes it, for the scripts that check or
measure it: `speech vad` with detection_flags() finds the speech alone, and padded() gives each stretch of speech the
audio the region keeps around it. It computes in the order RegionCut does, so that its seconds are the same doubles.
"""


def detection_flags(threshold, prefix_padding_ms, silence_duration_ms, max_speech_s):
    """The flags of `speech vad` that find the speech of transcription by regions with these values of server_vad's
    members and the longest region: no padding, and the longest region less the padding."""
    before, after = prefix_padding_ms / 1000, min(prefix_padding_ms, silence_duration_ms) / 1000
    return ["--threshold", repr(threshold), "--speech-pad-ms", "0", "--min-silence-duration-ms", str(silence_duration_ms),
            "--max-speech-duration-s", repr(max_speech_s - before - after)]


def padded(speech, prefix_padding_ms, silence_duration_ms, length):
    """The regions of `speech`, the (start, end) in seconds that `speech vad` with detection_flags() found in `length`
    seconds of audio: prefix_padding_ms before each, but not before the region before it ends, and as much after it, but
    no more than silence_duration_ms, or halfway to the speech after it where that starts within silence_duration_ms."""
    before, after = prefix_padding_ms / 1000, min(prefix_padding_ms, silence_duration_ms) / 1000
    silence = silence_duration_ms / 1000
    out, previous = [], 0.0
    for k, (start, end) in enumerate(speech):
        last = end + after
        if k + 1 < len(speech) and speech[k + 1][0] - end < silence:
            last = min(last, end + (speech[k + 1][0] - end) / 2)
        out.append((max(start - before, previous), min(last, length)))
        previous = out[-1][1]
    return out
