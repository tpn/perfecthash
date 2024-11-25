# Define the output CSV file
$outputFile = "system_info.csv"

# Initialize a hashtable to hold system information
$systemInfo = @{}

# Get computer name
$systemInfo["ComputerName"] = (Get-CimInstance Win32_ComputerSystem).Name

# Get CPU information
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$systemInfo["CpuName"] = $cpu.Name
$systemInfo["Architecture"] = $cpu.AddressWidth -eq 64 ? "x64" : "x86"
$systemInfo["NumberOfSockets"] = $cpu.SocketDesignation
$systemInfo["CoresPerSocket"] = $cpu.NumberOfCores
$systemInfo["BaseFrequency"] = "{0:N2}" -f ($cpu.MaxClockSpeed / 1000) # GHz
$systemInfo["MaxBoostFrequency"] = "" # Not readily available in WMI

# Initialize cache sizes
$l1CacheSize = ""
$l2CacheSize = ""
$l3CacheSize = ""

# Process cache memory information
$cacheData = Get-CimInstance Win32_CacheMemory

foreach ($cache in $cacheData) {
    switch ($cache.Level) {
        3 { $l1CacheSize = "{0} KB" -f $cache.MaxCacheSize }
        4 { $l2CacheSize = "{0} KB" -f $cache.MaxCacheSize }
        5 { $l3CacheSize = "{0} KB" -f $cache.MaxCacheSize }
    }
}

$systemInfo["L1CacheSize"] = $l1CacheSize
$systemInfo["L2CacheSize"] = $l2CacheSize
$systemInfo["L3CacheSize"] = $l3CacheSize
$systemInfo["LLCacheSize"] = $l3CacheSize # Logical Last-Level Cache matches L3 Cache

# Get memory size in GB
$memory = Get-CimInstance Win32_PhysicalMemory
$totalMemory = ($memory | Measure-Object -Property Capacity -Sum).Sum
$systemInfo["MemoryInGB"] = "{0:N2}" -f ($totalMemory / 1GB)

# Convert the hashtable to a CSV format and write it to the output file
$output = @(
    [pscustomobject]$systemInfo
)

# Write the CSV with PascalCase header
$output | Export-Csv -Path $outputFile -NoTypeInformation

# Display the location of the CSV file
Write-Host "System information written to $outputFile"

