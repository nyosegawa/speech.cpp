<#
Installs speech, speech.cpp's executable, from a release on GitHub on Windows x64, or updates it when run again. It
downloads the release's archive, checks it against the release's SHA256SUMS, unpacks it into
%LOCALAPPDATA%\Programs\speech.cpp\<version>\, checks that speech.exe starts, points the junction
%LOCALAPPDATA%\Programs\speech.cpp\current at it, and removes the other versions there. When that folder is not on the
user's PATH, it adds it and says so, unless -NoModifyPath is given. Models are not installed: speech fetches one the
first time a command names it.

usage: irm https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.ps1 | iex
       & ([scriptblock]::Create((irm https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.ps1))) [-Version X.Y.Z] [-NoModifyPath]
#>
param(
    [string] $Version = '',
    [switch] $NoModifyPath
)

# The body runs in a scope of its own, so that `irm | iex` leaves the session's preferences as they were.
& {
    # A failure throws rather than exits, since `irm | iex` runs this in the user's own session, which exit would close.
    $ErrorActionPreference = 'Stop'
    # Invoke-WebRequest's progress bar slows Windows PowerShell 5.1's downloads many times over.
    $ProgressPreference = 'SilentlyContinue'
    Set-StrictMode -Version 3
    # Windows PowerShell 5.1 on .NET Framework before 4.7 offers no TLS 1.2 unless asked, and GitHub requires it.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $Repository = 'nyosegawa/speech.cpp'

    function Stop-Install([string] $Message) {
        throw "speech.cpp installer: $Message"
    }

    # A 32-bit PowerShell reports x86 in PROCESSOR_ARCHITECTURE and the machine's in PROCESSOR_ARCHITEW6432.
    $architecture = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    if ($architecture -ne 'AMD64') {
        Stop-Install "the Windows release is for x64, and this machine is $architecture; build speech.cpp from source as its README says"
    }
    # speech.exe links the Vulkan loader, which every GPU driver of NVIDIA, AMD and Intel installs; without it Windows
    # refuses to start speech.exe with a dialog. A 32-bit process sees System32 through Sysnative.
    $system = if ([Environment]::Is64BitProcess) { 'System32' } else { 'Sysnative' }
    if (-not (Test-Path -LiteralPath (Join-Path $env:SystemRoot "$system\vulkan-1.dll"))) {
        Stop-Install 'speech.exe needs the Vulkan loader, vulkan-1.dll, which the GPU driver of NVIDIA, AMD or Intel installs; install the GPU''s driver and run this again'
    }

    if (-not $Version) {
        $Version = (Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$Repository/releases/latest").tag_name -replace '^v', ''
    }
    if ($Version -notmatch '^\d+\.\d+\.\d+$') {
        Stop-Install "`"$Version`" is not a release's version, such as 0.8.2"
    }

    $root = Join-Path $env:LOCALAPPDATA 'Programs\speech.cpp'
    $current = Join-Path $root 'current'
    $folder = Join-Path $root $Version
    $link = Get-Item -LiteralPath $current -Force -ErrorAction SilentlyContinue
    if ($link -and $link.LinkType -ne 'Junction') {
        Stop-Install "$current is there and is not this installer's junction; move it away and run this again"
    }

    $installed = ''
    if ($link -and (Test-Path -LiteralPath (Join-Path $current 'speech.exe'))) {
        $installed = (& (Join-Path $current 'speech.exe') --version) -join ''
    }
    if ($installed -like "speech.cpp $Version,*") {
        Write-Host "speech.cpp $Version is installed already, in $folder."
    } else {
        $archive = "speech-$Version-windows-x64-vulkan.zip"
        $base = "https://github.com/$Repository/releases/download/v$Version"
        # The work folder is under the install folder, so that the checked files move into place on the same volume.
        $createdRoot = -not (Test-Path -LiteralPath $root)
        $work = Join-Path $root ('.install-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $work | Out-Null
        $done = $false
        try {
            Write-Host "Downloading $base/$archive"
            Invoke-WebRequest -UseBasicParsing -Uri "$base/$archive" -OutFile (Join-Path $work $archive)
            Invoke-WebRequest -UseBasicParsing -Uri "$base/SHA256SUMS" -OutFile (Join-Path $work 'SHA256SUMS')
            $expected = ''
            foreach ($line in Get-Content -LiteralPath (Join-Path $work 'SHA256SUMS')) {
                $fields = @($line -split '\s+', 2)
                if ($fields.Count -eq 2 -and $fields[1].TrimStart('*') -eq $archive) { $expected = $fields[0].ToLowerInvariant() }
            }
            if (-not $expected) { Stop-Install "the release's SHA256SUMS lists no $archive" }
            $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $work $archive)).Hash.ToLowerInvariant()
            if ($actual -ne $expected) {
                Stop-Install "the SHA-256 of $archive is $actual, not $expected as the release's SHA256SUMS says; nothing was installed"
            }
            $unpacked = Join-Path $work 'unpacked'
            # Expand-Archive draws a progress bar per file in Windows PowerShell 5.1, which slows it many times over, and as a
            # function of a script module it does not see this scope's $ProgressPreference; .NET's ZipFile draws none.
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            [IO.Compression.ZipFile]::ExtractToDirectory((Join-Path $work $archive), $unpacked)
            $started = (& (Join-Path $unpacked 'speech.exe') --version) -join ''
            if ($LASTEXITCODE -ne 0 -or $started -notlike "speech.cpp $Version,*") {
                Stop-Install "the unpacked speech.exe says `"$started`" (exit code $LASTEXITCODE), not speech.cpp $Version"
            }
            if (Test-Path -LiteralPath $folder) { Remove-Item -LiteralPath $folder -Recurse -Force }
            Move-Item -LiteralPath $unpacked -Destination $folder
            # Removing a junction with Directory.Delete() removes the link alone; Remove-Item -Recurse in Windows PowerShell
            # 5.1 would empty the folder it points to.
            if ($link) { [IO.Directory]::Delete($current) }
            New-Item -ItemType Junction -Path $current -Target $folder | Out-Null
            $done = $true
        } finally {
            Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
            if (-not $done -and $createdRoot) { Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue }
        }
        foreach ($old in @(Get-ChildItem -LiteralPath $root -Directory | Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' -and $_.Name -ne $Version })) {
            try {
                Remove-Item -LiteralPath $old.FullName -Recurse -Force
                Write-Host "Removed $($old.Name), which $Version replaces."
            } catch {
                Write-Warning "Cannot remove $($old.FullName), which a running speech.exe may hold; remove it once that has ended."
            }
        }
        Write-Host "Installed speech.cpp $Version (windows-x64-vulkan) in $folder, which $current points to."
    }

    # The user's PATH is read and written in the registry as it is stored, so that entries such as %USERPROFILE%\bin stay
    # unexpanded and the value keeps its kind.
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
    try {
        $raw = [string] $key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $entries = @($raw -split ';' | Where-Object { $_ })
        $onPath = @($entries | Where-Object { [Environment]::ExpandEnvironmentVariables($_).TrimEnd('\') -eq $current }).Count -gt 0
        if ($onPath) {
            Write-Host 'Run: speech --help'
        } elseif ($NoModifyPath) {
            Write-Host "$current is not on your PATH, and -NoModifyPath changed nothing. Add it to your user PATH, or run $current\speech.exe directly."
        } else {
            $kind = if ($raw) { $key.GetValueKind('Path') } else { [Microsoft.Win32.RegistryValueKind]::ExpandString }
            $key.SetValue('Path', ((@($current) + $entries) -join ';'), $kind)
            # Programs that are running, Explorer among them, read the new PATH once told that the environment changed.
            if (-not ('SpeechCpp.Environment' -as [type])) {
                Add-Type -Namespace SpeechCpp -Name Environment -MemberDefinition @'
[DllImport("user32.dll", CharSet = CharSet.Unicode)]
public static extern IntPtr SendMessageTimeout(IntPtr window, uint message, UIntPtr wParam, string lParam, uint flags, uint timeout, out UIntPtr result);
'@
            }
            $result = [UIntPtr]::Zero
            [SpeechCpp.Environment]::SendMessageTimeout([IntPtr] 0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 2, 5000, [ref] $result) | Out-Null
            $env:Path = "$current;$env:Path"
            Write-Host "Added $current to the start of your user PATH (HKCU\Environment). This window has it already, and windows"
            Write-Host 'opened from now on will. Run: speech --help'
        }
    } finally {
        $key.Close()
    }
}
