#!/bin/bash

# Define output file
output_file="system_info.csv"

# Initialize variables
computer_name=$(scutil --get ComputerName)
cpu_name=$(sysctl -n machdep.cpu.brand_string)
architecture=$(uname -m)
number_of_sockets=$(sysctl -n hw.packages)
cores_per_socket=$(sysctl -n hw.physicalcpu)
base_frequency=$(sysctl -n hw.cpufrequency | awk '{printf "%.2f", $1 / 1000000000}') # GHz
max_boost_frequency="" # Not directly accessible on macOS
l1_cache_size=$(sysctl -n hw.l1dcachesize | awk '{printf "%.0f KB", $1 / 1024}')
l2_cache_size=$(sysctl -n hw.l2cachesize | awk '{printf "%.0f KB", $1 / 1024}')
l3_cache_size=$(sysctl -n hw.l3cachesize | awk '{printf "%.0f KB", $1 / 1024}')

# Logical last-level cache size (LLCacheSize matches L3CacheSize)
ll_cache_size="$l3_cache_size"

# Get total memory in GB
memory_in_gb=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1 / 1073741824}')

# Create the CSV header
echo "ComputerName,CpuName,Architecture,NumberOfSockets,CoresPerSocket,BaseFrequency,MaxBoostFrequency,L1CacheSize,L2CacheSize,L3CacheSize,LLCacheSize,MemoryInGB" > "$output_file"

# Add the system information to the CSV
echo "$computer_name,$cpu_name,$architecture,$number_of_sockets,$cores_per_socket,$base_frequency,$max_boost_frequency,$l1_cache_size,$l2_cache_size,$l3_cache_size,$ll_cache_size,$memory_in_gb" >> "$output_file"

# Print output file location
echo "System information written to $output_file"

