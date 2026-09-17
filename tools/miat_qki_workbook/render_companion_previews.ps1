param(
    [Parameter(Mandatory = $true)][string]$AnalysisWorkbook,
    [Parameter(Mandatory = $true)][string]$SpotsWorkbook,
    [Parameter(Mandatory = $true)][string]$NucleusWorkbook,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string]$PythonExecutable = 'C:\Users\ambur\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)

$ErrorActionPreference = 'Stop'

function Release-ComObject {
    param([object]$Object)
    if ($null -ne $Object) {
        try { [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($Object) } catch {}
    }
}

function Safe-FileStem {
    param([string]$Value)
    return (($Value.ToLowerInvariant() -replace '[^a-z0-9]+', '_').Trim('_'))
}

function Get-PreviewRangeAddress {
    param(
        [object]$Worksheet,
        [int]$MaximumRows,
        [int]$MaximumColumns
    )
    $used = $null
    $topLeft = $null
    $bottomRight = $null
    $range = $null
    try {
        $used = $Worksheet.UsedRange
        $rows = [Math]::Max(1, [Math]::Min($MaximumRows, [int]$used.Rows.Count))
        $columns = [Math]::Max(1, [Math]::Min($MaximumColumns, [int]$used.Columns.Count))
        $topLeft = $Worksheet.Cells.Item(1, 1)
        $bottomRight = $Worksheet.Cells.Item($rows, $columns)
        $range = $Worksheet.Range($topLeft, $bottomRight)
        return [string]$range.Address($false, $false)
    }
    finally {
        Release-ComObject $range
        Release-ComObject $bottomRight
        Release-ComObject $topLeft
        Release-ComObject $used
    }
}

function Export-WorksheetPreview {
    param(
        [object]$Worksheet,
        [string]$RangeAddress,
        [string]$PdfPath,
        [string]$PngBasePath,
        [string]$PdfToPpm
    )
    $pageSetup = $null
    try {
        $pageSetup = $Worksheet.PageSetup
        $pageSetup.PrintArea = $RangeAddress
        $pageSetup.Orientation = 2
        $pageSetup.Zoom = $false
        $pageSetup.FitToPagesWide = 1
        $pageSetup.FitToPagesTall = 1
        $pageSetup.CenterHorizontally = $false
        $pageSetup.CenterVertically = $false
        [void]$Worksheet.ExportAsFixedFormat(0, $PdfPath, 0, $true, $true)
        if (-not (Test-Path -LiteralPath $PdfPath) -or (Get-Item -LiteralPath $PdfPath).Length -le 0) {
            throw "Excel exported an empty PDF: $PdfPath"
        }

        & $PdfToPpm '-png' '-f' '1' '-singlefile' '-r' '200' $PdfPath $PngBasePath | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "pdftoppm failed with exit code $LASTEXITCODE for $PdfPath"
        }
        $pngPath = "$PngBasePath.png"
        if (-not (Test-Path -LiteralPath $pngPath) -or (Get-Item -LiteralPath $pngPath).Length -le 5000) {
            throw "Rasterized preview is missing or unexpectedly small: $pngPath"
        }
        return [pscustomobject]@{
            sheet = [string]$Worksheet.Name
            range = $RangeAddress
            pdf = $PdfPath
            pdfBytes = (Get-Item -LiteralPath $PdfPath).Length
            png = $pngPath
            pngBytes = (Get-Item -LiteralPath $pngPath).Length
        }
    }
    finally {
        Release-ComObject $pageSetup
    }
}

function Export-AnalysisPreviews {
    param(
        [object]$Excel,
        [string]$WorkbookPath,
        [string]$PdfDirectory,
        [string]$PngDirectory,
        [string]$PdfToPpm
    )
    $book = $null
    $outputs = [System.Collections.Generic.List[object]]::new()
    try {
        $book = $Excel.Workbooks.Open($WorkbookPath, 0, $true)
        foreach ($sheet in @($book.Worksheets)) {
            try {
                if ($sheet.Name -eq '_Chart Data') { continue }
                $stem = "analysis_$(Safe-FileStem $sheet.Name)"
                if ($sheet.Name -eq 'Summary Charts') {
                    $rangeAddress = 'A1:Z34'
                }
                elseif ($sheet.Name -eq 'Summary') {
                    $rangeAddress = 'A1:P10'
                }
                elseif ($sheet.Name -eq 'README') {
                    $rangeAddress = Get-PreviewRangeAddress -Worksheet $sheet -MaximumRows 24 -MaximumColumns 2
                }
                elseif ($sheet.Name -eq 'Data Dictionary') {
                    $rangeAddress = Get-PreviewRangeAddress -Worksheet $sheet -MaximumRows 28 -MaximumColumns 6
                }
                else {
                    $rangeAddress = Get-PreviewRangeAddress -Worksheet $sheet -MaximumRows 28 -MaximumColumns 12
                }
                $pdfPath = Join-Path $PdfDirectory "$stem.pdf"
                $pngBase = Join-Path $PngDirectory $stem
                $outputs.Add((Export-WorksheetPreview -Worksheet $sheet -RangeAddress $rangeAddress -PdfPath $pdfPath -PngBasePath $pngBase -PdfToPpm $PdfToPpm))
            }
            finally {
                Release-ComObject $sheet
            }
        }
        return $outputs
    }
    finally {
        if ($null -ne $book) { $book.Close($false) }
        Release-ComObject $book
    }
}

function Export-RawPreview {
    param(
        [object]$Excel,
        [string]$WorkbookPath,
        [string]$SheetName,
        [string]$Stem,
        [string]$PdfDirectory,
        [string]$PngDirectory,
        [string]$PdfToPpm
    )
    $book = $null
    $sheet = $null
    try {
        $book = $Excel.Workbooks.Open($WorkbookPath, 0, $true)
        $sheet = $book.Worksheets.Item($SheetName)
        return Export-WorksheetPreview -Worksheet $sheet -RangeAddress 'A1:L25' -PdfPath (Join-Path $PdfDirectory "$Stem.pdf") -PngBasePath (Join-Path $PngDirectory $Stem) -PdfToPpm $PdfToPpm
    }
    finally {
        Release-ComObject $sheet
        if ($null -ne $book) { $book.Close($false) }
        Release-ComObject $book
    }
}

$analysisPath = (Resolve-Path -LiteralPath $AnalysisWorkbook).Path
$spotsPath = (Resolve-Path -LiteralPath $SpotsWorkbook).Path
$nucleusPath = (Resolve-Path -LiteralPath $NucleusWorkbook).Path
$destination = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $destination) {
    throw "Preview output directory already exists; refusing overwrite: $destination"
}
$pdfToPpm = (Get-Command pdftoppm -ErrorAction Stop).Source
[void](New-Item -ItemType Directory -Path $destination)
$pdfDirectory = Join-Path $destination 'pdf'
$pngDirectory = Join-Path $destination 'png'
[void](New-Item -ItemType Directory -Path $pdfDirectory)
[void](New-Item -ItemType Directory -Path $pngDirectory)

$excel = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.ScreenUpdating = $false
    $excel.EnableEvents = $false
    $excel.AskToUpdateLinks = $false

    $outputs = [System.Collections.Generic.List[object]]::new()
    foreach ($item in (Export-AnalysisPreviews -Excel $excel -WorkbookPath $analysisPath -PdfDirectory $pdfDirectory -PngDirectory $pngDirectory -PdfToPpm $pdfToPpm)) {
        $outputs.Add($item)
    }
    $rawRenderer = Join-Path $PSScriptRoot 'render_raw_first_pages.py'
    $rawDirectory = Join-Path $destination 'raw'
    $rawJson = & $PythonExecutable $rawRenderer '--spots-workbook' $spotsPath '--nucleus-workbook' $nucleusPath '--output-dir' $rawDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Bounded raw first-page renderer failed: $rawJson"
    }
    $rawReport = $rawJson | ConvertFrom-Json
    foreach ($preview in $rawReport.previews) {
        $outputs.Add([pscustomobject]@{
            sheet = $preview.sheet
            range = 'A1:L25'
            pdf = $null
            pdfBytes = $null
            png = $preview.png
            pngBytes = $preview.pngBytes
        })
    }

    $report = [pscustomobject]@{
        status = 'success'
        renderer = 'Microsoft Excel ExportAsFixedFormat + Poppler pdftoppm'
        outputDirectory = $destination
        previewCount = $outputs.Count
        previews = $outputs
    }
    $json = $report | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText((Join-Path $destination 'preview_manifest.json'), $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    Write-Output ($json -replace '\r?\n', '')
}
finally {
    if ($null -ne $excel) {
        try { $excel.Quit() } catch {}
    }
    Release-ComObject $excel
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
