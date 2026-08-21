#!/usr/bin/env bash

# Run file discovery followed by one detection pipeline per SD card/site pair.
# All paths are resolved from the directory where this script is invoked.

set -Eeuo pipefail

readonly SCRIPT_NAME="${0##*/}"
active_pid=""

usage() {
    echo "Usage: $SCRIPT_NAME [--skip-file-dealer] RECOVERY_CONFIG" >&2
    echo "" >&2
    echo "Options:" >&2
    echo "  -s, --skip-file-dealer  Skip indexing recordings with src/file_dealer.py" >&2
    echo "  -h, --help              Show this help message" >&2
}

die() {
    echo "$SCRIPT_NAME: $*" >&2
    exit 1
}

stop_active_process() {
    local exit_status=$?

    if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
        echo "$SCRIPT_NAME: stopping process $active_pid" >&2
        kill "$active_pid" 2>/dev/null || true
        wait "$active_pid" 2>/dev/null || true
    fi

    exit "$exit_status"
}

run_and_wait() {
    local description=$1
    local process_status
    shift

    echo "Starting: $description"
    nohup "$@" &
    active_pid=$!
    echo "Process ID: $active_pid"

    if wait "$active_pid"; then
        active_pid=""
        echo "Completed: $description"
    else
        process_status=$?
        active_pid=""
        echo "$SCRIPT_NAME: '$description' failed with exit status $process_status; no remaining jobs will run." >&2
        return "$process_status"
    fi
}

skip_file_dealer=false
config_file=""

while (( $# > 0 )); do
    case $1 in
        -s|--skip-file-dealer)
            skip_file_dealer=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            (( $# > 0 )) || die "missing recovery configuration file"
            [[ -z "$config_file" ]] || die "only one recovery configuration file may be provided"
            config_file=$1
            shift
            (( $# == 0 )) || die "unexpected argument: $1"
            break
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            [[ -z "$config_file" ]] || die "only one recovery configuration file may be provided"
            config_file=$1
            ;;
    esac
    shift
done

if [[ -z "$config_file" ]]; then
    usage
    exit 2
fi

readonly config_file
[[ -r "$config_file" ]] || die "cannot read configuration file: $config_file"
if [[ $skip_file_dealer == false ]]; then
    [[ -f src/file_dealer.py ]] || die "src/file_dealer.py was not found; run this script from the repository root"
fi
[[ -f src/batdt2_pipeline.py ]] || die "src/batdt2_pipeline.py was not found; run this script from the repository root"

mount_path=""
recover_folder=""
declare -a sd_units=()
declare -a sites=()
config_item=0
line_number=0

# Validate the entire configuration before starting any long-running work.
while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))
    line=${line%$'\r'}
    [[ $line =~ ^[[:space:]]*$ ]] && continue

    config_item=$((config_item + 1))
    if (( config_item <= 2 )); then
        if [[ $line =~ ^[[:space:]]*\"([^\"]+)\"[[:space:]]*$ ]]; then
            if (( config_item == 1 )); then
                mount_path=${BASH_REMATCH[1]}
            else
                recover_folder=${BASH_REMATCH[1]}
            fi
        else
            die "invalid line $line_number: expected one double-quoted value"
        fi
    elif [[ $line =~ ^[[:space:]]*\"([^\"]+)\"[[:space:]]+\"([^\"]+)\"[[:space:]]*$ ]]; then
        sd_units+=("${BASH_REMATCH[1]}")
        sites+=("${BASH_REMATCH[2]}")
    else
        die "invalid line $line_number: expected \"SD_UNIT\" \"SITE NAME\""
    fi
done < "$config_file"

[[ -n "$mount_path" ]] || die "configuration is missing the input directory"
[[ -n "$recover_folder" ]] || die "configuration is missing the recovery folder"
(( ${#sd_units[@]} > 0 )) || die "configuration has no SD unit/site entries"

trap stop_active_process INT TERM HUP

mount_name=${mount_path%/}
mount_name=${mount_name##*/}
[[ -n "$mount_name" ]] || die "cannot derive a name from input directory: $mount_path"

if [[ $skip_file_dealer == false ]]; then
    file_dealer_command=(
        python src/file_dealer.py
        "$mount_path"
        "output_dir"
        "${mount_name}_collected_audio_records.csv"
    )
    run_and_wait "index recordings in $mount_path" "${file_dealer_command[@]}"
else
    echo "Skipping file_dealer.py as requested."
fi

for ((i = 0; i < ${#sd_units[@]}; i++)); do
    detection_command=(
        python3 src/batdt2_pipeline.py
        --recover_folder="$recover_folder"
        --sd_unit="${sd_units[i]}"
        --site="${sites[i]}"
        --recording_start="00:00"
        --recording_end="23:59"
        --cycle_length=600
        --duration=300
        --output_directory="output_dir/$recover_folder"
        --num_processes=4
        --run_model
        --generate_fig
        --csv
    )
    run_and_wait "detect SD unit ${sd_units[i]} at ${sites[i]}" "${detection_command[@]}"
done

echo "All detection pipelines completed successfully."
