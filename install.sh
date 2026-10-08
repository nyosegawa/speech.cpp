#!/bin/sh
# Installs speech, speech.cpp's executable, from a release on GitHub on macOS (Apple silicon) and Linux (x86-64), or
# updates it when run again. It picks the release's archive for the system and GPU, checks it against the release's
# SHA256SUMS, unpacks it into ~/.local/share/speech.cpp/<version>/, checks that speech starts, links
# ~/.local/bin/speech to it and removes the other versions there. When ~/.local/bin is not on PATH, it adds it in the
# shell's startup file and says which, unless --no-modify-path is given. Models are not installed: speech fetches one
# the first time a command names it.
#
# usage: curl -fsSL https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh | sh
#        curl -fsSL https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh | sh -s -- [options]
#   --version X.Y.Z     install this release rather than the latest
#   --no-modify-path    change no startup file

set -eu

repository=nyosegawa/speech.cpp
version=
modify_path=yes

fail() {
    printf 'speech.cpp installer: %s\n' "$1" >&2
    exit 1
}

say() {
    printf '%s\n' "$1" >&2
}

# A path as the messages write it, with ~ for the home folder.
home() {
    case "$1" in
        "$HOME"/*) printf '~/%s' "${1#"$HOME"/}" ;;
        *) printf '%s' "$1" ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version)
            [ $# -ge 2 ] || fail "--version needs a release, such as 0.8.2"
            version=$2
            shift 2
            ;;
        --version=*)
            version=${1#*=}
            shift
            ;;
        --no-modify-path)
            modify_path=no
            shift
            ;;
        -h | --help)
            say "usage: install.sh [--version X.Y.Z] [--no-modify-path]"
            say "Installs speech.cpp's latest release, or X.Y.Z, into $HOME/.local/share/speech.cpp, links $HOME/.local/bin/speech"
            say "to it, and adds $HOME/.local/bin to PATH in the shell's startup file unless --no-modify-path is given."
            exit 0
            ;;
        *)
            fail "there is no option $1; the options are --version X.Y.Z and --no-modify-path"
            ;;
    esac
done

command -v curl >/dev/null 2>&1 || fail "curl is needed to download the release; install it and run this again"
command -v unzip >/dev/null 2>&1 || fail "unzip is needed to unpack the release; install it (apt install unzip, dnf install unzip) and run this again"

# The archive for this system: a release has one for macOS arm64 with Metal, and for Linux x86-64 one with Vulkan and
# one for the CPU alone.
case "$(uname -s)" in
    Darwin)
        # A shell that runs under Rosetta says x86_64 on an Apple silicon Mac, which runs the arm64 build all the same.
        if [ "$(uname -m)" = arm64 ] || [ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" = 1 ]; then
            platform=macos-arm64-metal
        else
            fail "the releases run on Apple silicon; on an Intel Mac, build speech.cpp from source as its README says"
        fi
        ;;
    Linux)
        [ "$(uname -m)" = x86_64 ] || fail "the Linux releases are for x86-64, not $(uname -m); build speech.cpp from source as its README says"
        glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | sed -n 's/^glibc \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
        [ -n "$glibc" ] || fail "the Linux releases need glibc 2.34 or later, and this system has no glibc (musl, as on Alpine?); build speech.cpp from source"
        set -- $glibc
        if [ "$1" -lt 2 ] || { [ "$1" -eq 2 ] && [ "$2" -lt 34 ]; }; then
            fail "the Linux releases need glibc 2.34 or later (Ubuntu 22.04, Debian 12, Fedora 35, RHEL 9), and this system has $1.$2"
        fi
        for flag in avx2 fma f16c; do
            grep -qw "$flag" /proc/cpuinfo || fail "the Linux releases need a CPU with AVX2, FMA and F16C, and this one has no $flag"
        done
        # The Vulkan build links libvulkan.so.1 and does not start without it; with the loader and no GPU driver it runs
        # on the CPU, since ggml registers no Vulkan device when no driver answers. The dynamic linker finds the loader
        # through its cache and LD_LIBRARY_PATH.
        loader=no
        for ldconfig in ldconfig /sbin/ldconfig /usr/sbin/ldconfig; do
            if command -v "$ldconfig" >/dev/null 2>&1; then
                if "$ldconfig" -p 2>/dev/null | grep -q 'libvulkan\.so\.1 (libc6,x86-64)'; then loader=yes; fi
                break
            fi
        done
        old_ifs=$IFS
        IFS=:
        for dir in ${LD_LIBRARY_PATH:-}; do
            if [ -n "$dir" ] && [ -e "$dir/libvulkan.so.1" ]; then loader=yes; fi
        done
        IFS=$old_ifs
        if [ "$loader" = yes ]; then
            platform=linux-x64-vulkan
        else
            platform=linux-x64-cpu
            say "The Vulkan loader, libvulkan.so.1, is not installed, so this installs the build for the CPU alone. For a GPU, install"
            say "the loader (libvulkan1 on Debian and Ubuntu, vulkan-loader on Fedora) and the GPU's Vulkan driver, and run this again."
        fi
        ;;
    *)
        fail "the releases are for macOS and Linux, not $(uname -s); on Windows run install.ps1, as speech.cpp's README says"
        ;;
esac

# The latest release is where GitHub's /releases/latest redirects, which needs no API and counts against no rate limit.
if [ -z "$version" ]; then
    latest=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$repository/releases/latest") ||
        fail "cannot reach github.com to find the latest release; check the network"
    version=${latest##*/v}
fi
case "$version" in
    [0-9]*.[0-9]*.[0-9]*) ;;
    *) fail "\"$version\" is not a release's version, such as 0.8.2" ;;
esac

root=$HOME/.local/share/speech.cpp
bin=$HOME/.local/bin
target=$root/$version/speech
link=$bin/speech
archive=speech-$version-$platform.zip
# The archive a version was installed from, which tells a build for the CPU from one for Vulkan of the same version.
record=$root/$version/.archive
if [ -e "$link" ] || [ -L "$link" ]; then
    case "$(readlink "$link" 2>/dev/null || true)" in
        "$root"/*) ;;
        *) fail "$(home "$link") is there and is not this installer's link; move it away and run this again" ;;
    esac
fi

installed=$(cat "$record" 2>/dev/null || true)
if [ "$(readlink "$link" 2>/dev/null || true)" = "$target" ] && [ -x "$target" ] && [ "$installed" = "$archive" ]; then
    say "speech.cpp $version is installed already, in $(home "$root/$version")."
else
    if [ -n "$installed" ] && [ "$installed" != "$archive" ]; then
        say "speech.cpp $version is installed from $installed; this system takes $archive, which replaces it."
    fi
    base=https://github.com/$repository/releases/download/v$version
    # Nothing goes under the home folder until the archive is checked and its speech starts.
    work=$(mktemp -d "${TMPDIR:-/tmp}/speech.cpp-install.XXXXXX")
    trap 'rm -rf "$work"' EXIT
    trap 'rm -rf "$work"; exit 1' INT TERM
    say "Downloading $base/$archive"
    curl -fsSL -o "$work/$archive" "$base/$archive" || fail "cannot download $base/$archive; check that release $version exists and has it"
    curl -fsSL -o "$work/SHA256SUMS" "$base/SHA256SUMS" || fail "cannot download $base/SHA256SUMS"
    expected=$(awk -v name="$archive" '$2 == name || $2 == "*" name { print $1 }' "$work/SHA256SUMS")
    [ -n "$expected" ] || fail "the release's SHA256SUMS lists no $archive"
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$work/$archive" | cut -d' ' -f1)
    else
        actual=$(shasum -a 256 "$work/$archive" | cut -d' ' -f1)
    fi
    [ "$actual" = "$expected" ] || fail "the SHA-256 of $archive is $actual, not $expected as the release's SHA256SUMS says; nothing was installed"
    mkdir "$work/unpacked"
    unzip -q "$work/$archive" -d "$work/unpacked" || fail "cannot unpack $archive"
    started=$("$work/unpacked/speech" --version 2>&1) || fail "the unpacked speech does not start: $started"
    case "$started" in
        "speech.cpp $version,"*) ;;
        *) fail "the unpacked speech says \"$started\", not speech.cpp $version" ;;
    esac
    printf '%s\n' "$archive" >"$work/unpacked/.archive"
    mkdir -p "$root" "$bin"
    rm -rf "${root:?}/$version"
    mv "$work/unpacked" "$root/$version"
    # The link is made beside the old one and renamed over it, so that speech is never missing from PATH.
    ln -sf "$target" "$bin/.speech.$$"
    mv -f "$bin/.speech.$$" "$link"
    for old in "$root"/[0-9]*.[0-9]*.[0-9]*; do
        if [ -d "$old" ] && [ "$old" != "$root/$version" ]; then
            rm -rf "$old"
            say "Removed ${old##*/}, which $version replaces."
        fi
    done
    say "Installed speech.cpp $version ($platform) in $(home "$root/$version"), linked from $(home "$link")."
fi

case ":${PATH:-}:" in
    *":$bin:"*)
        say "Run: speech --help"
        exit 0
        ;;
esac
line='export PATH="$HOME/.local/bin:$PATH"'
case "${SHELL:-}" in
    */zsh) profile=${ZDOTDIR:-$HOME}/.zshrc ;;
    */bash)
        # Terminal on macOS starts login shells, which read .bash_profile; terminals on Linux start shells that read .bashrc.
        if [ "$(uname -s)" = Darwin ]; then profile=$HOME/.bash_profile; else profile=$HOME/.bashrc; fi
        ;;
    */fish)
        profile=$HOME/.config/fish/conf.d/speech.cpp.fish
        line='fish_add_path -g "$HOME/.local/bin"'
        ;;
    *) profile=$HOME/.profile ;;
esac
if [ "$modify_path" = no ]; then
    say "$(home "$bin") is not on PATH, and --no-modify-path changed nothing. Add it with this line in your shell's startup file:"
    say "  $line"
    say "or run $(home "$link") directly."
elif [ -f "$profile" ] && grep -qxF "$line" "$profile"; then
    say "$(home "$profile") adds $(home "$bin") to PATH already; open a new terminal, then run: speech --help"
else
    mkdir -p "$(dirname "$profile")"
    printf '\n# Added by speech.cpp'"'"'s installer\n%s\n' "$line" >>"$profile"
    say "Added $(home "$bin") to PATH in $(home "$profile"), with the line"
    say "  $line"
    say "Open a new terminal, or run that line, then run: speech --help"
fi
