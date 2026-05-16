# ============================================================
# Create_Colab_Zip.ps1
# Zips the SkyLogic project SOURCE CODE only (no venv, no data).
# Upload the resulting skylogic_code.zip to your Google Drive root.
# ============================================================

$ProjectRoot = $PSScriptRoot   # same directory as this script
$OutputZip   = Join-Path $ProjectRoot "skylogic_code.zip"

Write-Host "Project root  : $ProjectRoot"
Write-Host "Output zip    : $OutputZip"

# Remove old zip if it exists
if (Test-Path $OutputZip) {
    Remove-Item $OutputZip -Force
    Write-Host "Removed old zip."
}

# Folders/files to EXCLUDE (venv internals, data, git, cache)
$Excludes = @(
    "Lib", "Scripts", "Include", "share",
    "data", "runs", ".git", ".vscode",
    "__pycache__", "*.pyc", "*.db",
    "pyvenv.cfg",
    "skylogic_code.zip",
    "*.zip"
)

# Collect items to include
$Items = Get-ChildItem -Path $ProjectRoot -Depth 0 | Where-Object {
    $_.Name -notin $Excludes -and
    $_.Name -notmatch "\.zip$" -and
    $_.Name -notmatch "\.db$"
}

Write-Host "`nItems to include:"
foreach ($item in $Items) {
    Write-Host "  + $($item.Name)"
}

# Use .NET ZipFile for reliable compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$ZipStream = [System.IO.Compression.ZipFile]::Open($OutputZip, 'Create')
$ZipStream.Dispose()

# Re-open for update
$ZipStream = [System.IO.Compression.ZipFile]::Open($OutputZip, 'Update')

function Add-ToZip {
    param($ZipArchive, $SourcePath, $EntryBase)

    if (Test-Path $SourcePath -PathType Leaf) {
        # Single file
        $entryName = $EntryBase + (Split-Path $SourcePath -Leaf)
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $ZipArchive, $SourcePath, $entryName,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    } elseif (Test-Path $SourcePath -PathType Container) {
        $DirName  = Split-Path $SourcePath -Leaf
        $BaseName = $EntryBase + $DirName + "/"

        $AllFiles = Get-ChildItem -Path $SourcePath -Recurse -File | Where-Object {
            # Skip venv internals when recursing into skylogic/
            $_.FullName -notmatch "[\\\/](__pycache__|Lib|Scripts|Include|share|\.git)[\\\/]" -and
            $_.FullName -notmatch "\.pyc$" -and
            $_.FullName -notmatch "pyvenv\.cfg"
        }

        foreach ($file in $AllFiles) {
            $relative  = $file.FullName.Substring($SourcePath.Length).TrimStart('\','/')
            $entryName = $BaseName + $relative.Replace('\', '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $ZipArchive, $file.FullName, $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
}

$FileCount = 0
foreach ($item in $Items) {
    $SkipPatterns = @("Lib", "Scripts", "Include", "share", "data", "runs", ".git", ".vscode")
    if ($item.Name -in $SkipPatterns) { continue }

    Write-Host "  Zipping: $($item.Name) ..."
    Add-ToZip -ZipArchive $ZipStream -SourcePath $item.FullName -EntryBase ""
    $FileCount++
}

$ZipStream.Dispose()

$ZipInfo = Get-Item $OutputZip
$SizeMB  = [math]::Round($ZipInfo.Length / 1MB, 2)

Write-Host "`n============================================"
Write-Host "ZIP CREATED SUCCESSFULLY"
Write-Host "============================================"
Write-Host "  Path : $OutputZip"
Write-Host "  Size : $SizeMB MB"
Write-Host ""
Write-Host "NEXT STEPS:"
Write-Host "  1. Upload '$OutputZip' to your Google Drive root"
Write-Host "     (it should appear at: My Drive/skylogic_code.zip)"
Write-Host "  2. Upload 'SkyLogic_Colab_GPU.ipynb' to Google Colab"
Write-Host "  3. Set Colab runtime to GPU (T4 or better)"
Write-Host "  4. Run all cells top-to-bottom"
Write-Host ""
