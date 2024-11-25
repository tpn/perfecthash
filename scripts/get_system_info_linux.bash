#!/bin/bash

# Define output file
output_file="system_info.csv"

# Initialize variables
computer_name=$(hostname)
cpu_info=$(lscpu)
cpu_name=$(echo "$cpu_info" | grep -m 1 "Model name" | awk -F: '{print $2}' | xargs)
architecture=$(echo "$cpu_info" | grep -m 1 "Architecture" | awk -F: '{print $2}' | xargs)
number_of_sockets=$(echo "$cpu_info" | grep -m 1 "Socket(s)" | awk -F: '{print $2}' | xargs)
cores_per_socket=$(echo "$cpu_info" | grep -m 1 "Core(s) per socket" | awk -F: '{print $2}' | xargs)
base_frequency=$(echo "$cpu_info" | grep -m 1 "CPU MHz" | awk -F: '{print $2}' | xargs)
l1_cache_size=$(echo "$cpu_info" | grep -m 1 "L1d cache" | awk -F: '{print $2}' | xargs)
l2_cache_size=$(echo "$cpu_info" | grep -m 1 "L2 cache" | awk -F: '{print $2}' | xargs)
l3_cache_size=$(echo "$cpu_info" | grep -m 1 "L3 cache" | awk -F: '{print $2}' | xargs)

# Logical last-level cache size (LLCacheSize matches L3CacheSize)
ll_cache_size="$l3_cache_size"

# MaxBoostFrequency is not easily accessible on Linux
max_boost_frequency=""

# Get total memory in GB
memory_in_gb=$(free -g | awk '/^Mem:/{print $2}')

# Create the CSV header
echo "ComputerName,CpuName,Architecture,NumberOfSockets,CoresPerSocket,BaseFrequency,MaxBoostFrequency,L1CacheSize,L2CacheSize,L3CacheSize,LLCacheSize,MemoryInGB" > "$output_file"

# Add the system information to the CSV
echo "$computer_name,$cpu_name,$architecture,$number_of_sockets,$cores_per_socket,$base_frequency,$max_boost_frequency,$l1_cache_size,$l2_cache_size,$l3_cache_size,$ll_cache_size,$memory_in_gb" >> "$output_file"

# Print output file location
echo "System information written to $output_file"

