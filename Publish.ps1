<#
.SYNOPSIS
    Publishes the current state of this folder to GitHub as a SINGLE commit,
    erasing all previous commit history on the remote.

.USAGE
    .\Publish.ps1
    .\Publish.ps1 -Message "Updated grader"
#>

param(
    [string]$Message = "Pokemon card pre-grader"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Build a fresh orphan branch containing only the current files
git checkout --orphan publish-temp
git add -A
git commit -m $Message

# Replace master with it
git branch -D master
git branch -m master

# Overwrite remote history
git push --force origin master

Write-Host ""
Write-Host "Published. Remote history is now a single commit: '$Message'" -ForegroundColor Green
git log --oneline
