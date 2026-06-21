#!/usr/bin/env bash
if [[ -z "${BASH_VERSION-}" ]]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# ====== VM Migration Script for GPU VMs ======
# This script migrates any VM to a zone with GPU availability
# by creating a disk snapshot and launching a new instance with GPUs
# Default: 4x A100 40GB GPUs (can be changed with MACHINE_TYPE/GPU_TYPE/GPU_COUNT)
# Usage: SOURCE_INSTANCE=<name> SOURCE_ZONE=<zone> ./launch_vm.sh
set -euo pipefail

usage() {
  cat <<'EOF'
VM Migration Script for GPU VMs

Modes:
  1) Migrate an existing VM (default):
     SOURCE_INSTANCE=<name> SOURCE_ZONE=<zone> ./launch_vm.sh

  2) Use an existing snapshot:
     SKIP_SNAPSHOT=true DISK_SNAPSHOT_NAME=<snapshot> SNAPSHOT_REGION=<region> ./launch_vm.sh

  3) Create a fresh GPU VM from a ML image:
     CREATE_FROM_IMAGE=true NEW_INSTANCE_NAME=<name> ./launch_vm.sh

Interactive behavior:
  If no inputs are provided, the script will prompt you to choose one of the
  three workflows above, then collect the required values interactively.

Key options (env vars):
  PROJECT_ID              GCP project (default: gcloud config)
  MACHINE_TYPE            e.g., a2-highgpu-4g (default: a2-highgpu-4g)
  GPU_TYPE                e.g., nvidia-h100-80gb (default: nvidia-h100-80gb)
  GPU_COUNT               e.g., 4 (auto-detected from machine type)

GPU / Machine type compatibility:
  nvidia-h100-*, nvidia-h200-*, nvidia-b200-*  → A3 machines (a3-highgpu-8g, etc.)
  nvidia-a100-*, nvidia-tesla-a100             → A2 machines (a2-highgpu-4g, etc.)
  nvidia-l4                                    → G2 machines (g2-standard-8, etc.)
  REGIONS                 space-separated list of regions to scan
                          Default: europe-west1 europe-west2 europe-west3
                                   europe-west4 europe-west8 europe-west9
                                   europe-central2 asia-northeast1
                                   asia-northeast2 asia-northeast3
                                   asia-east1 asia-east2 asia-southeast1
                                   asia-southeast2 us-central1 us-east1
                                   us-east4 us-west1 us-west2 us-west4

Snapshot options:
  SNAPSHOT_PREFIX          Prefix for new snapshots (default: jeff)
  SKIP_SNAPSHOT            true/false (default: false)

New VM from image options:
  CREATE_FROM_IMAGE        true/false (default: false)
  ML_IMAGE_PROJECT         Default: deeplearning-platform-release
  ML_IMAGE_FAMILY          Default: pytorch-2-7-cu128-ubuntu-2404-nvidia-570
  BOOT_DISK_SIZE_GB         Default: 100

Networking overrides (optional):
  NETWORK, SUBNET, TAGS, SERVICE_ACCOUNT, SCOPES (default: empty)

Reservations:
  SKIP_RESERVATIONS         true/false (default: true)
  DELETE_RESERVATION_AFTER  true/false (default: true)

Examples:
  SOURCE_INSTANCE=my-vm SOURCE_ZONE=us-east1-b ./launch_vm.sh
  SKIP_SNAPSHOT=true DISK_SNAPSHOT_NAME=my-snap SNAPSHOT_REGION=us-central1 ./launch_vm.sh
  CREATE_FROM_IMAGE=true NEW_INSTANCE_NAME=my-gpu-vm ./launch_vm.sh
EOF
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
  exit 0
fi

prompt_required() {
  local label="$1"
  local default="${2-}"
  local value=""
  while [[ -z "$value" ]]; do
    if [[ -n "$default" ]]; then
      read -r -p "${label} [${default}]: " value
      value="${value:-$default}"
    else
      read -r -p "${label}: " value
    fi
  done
  echo "$value"
}

select_from_list() {
  local label="$1"
  shift
  local options=("$@")
  local choice=""

  if [[ ${#options[@]} -gt 0 ]]; then
    printf '%s\n' "$label" >&2
    for i in "${!options[@]}"; do
      printf '  %s) %s\n' "$((i + 1))" "${options[$i]}" >&2
    done
    read -r -p "Select [1-${#options[@]}], or type a name: " choice
    if [[ -z "$choice" ]]; then
      echo "${options[0]}"
      return
    fi
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
      echo "${options[$((choice - 1))]}"
      return
    fi
    echo "$choice"
    return
  fi

  echo "$(prompt_required "$label")"
}

select_from_list_with_default() {
  local label="$1"
  local default="$2"
  shift 2
  local options=("$@")
  local choice=""

  if [[ ${#options[@]} -gt 0 ]]; then
    printf '%s\n' "$label" >&2
    for i in "${!options[@]}"; do
      printf '  %s) %s\n' "$((i + 1))" "${options[$i]}" >&2
    done
    if [[ -n "$default" ]]; then
      read -r -p "Select [1-${#options[@]}] (default: ${default}), or select a number: " choice
      if [[ -z "$choice" ]]; then
        echo "$default"
        return
      fi
    else
      read -r -p "Select [1-${#options[@]}], or select a number: " choice
      if [[ -z "$choice" ]]; then
        echo "${options[0]}"
        return
      fi
    fi
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
      echo "${options[$((choice - 1))]}"
      return
    fi
    echo "$choice"
    return
  fi

  echo "$(prompt_required "$label" "$default")"
}

pytorch_version_ok() {
  local family="$1"
  if [[ "$family" =~ pytorch-([0-9]+)-([0-9]+) ]]; then
    local major="${BASH_REMATCH[1]}"
    local minor="${BASH_REMATCH[2]}"
    if (( major > 2 )); then
      return 0
    fi
    if (( major == 2 && minor >= 1 )); then
      return 0
    fi
  fi
  return 1
}

# ====== REQUIRED: identify your source instance ======
SOURCE_INSTANCE="${SOURCE_INSTANCE:-}"           # e.g. "my-training-vm"
SOURCE_ZONE="${SOURCE_ZONE:-}"                   # e.g. "us-east1-b"
NEW_INSTANCE_NAME="${NEW_INSTANCE_NAME:-}"
DISK_SNAPSHOT_NAME="${DISK_SNAPSHOT_NAME:-}"
SNAPSHOT_REGION="${SNAPSHOT_REGION:-}"

# ====== OPTIONAL: tweak behavior ======
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
DEFAULT_COMPUTE_ZONE="$(gcloud config get-value compute/zone 2>/dev/null)"
DEFAULT_COMPUTE_REGION="$(gcloud config get-value compute/region 2>/dev/null)"
GCLOUD_ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-4g}"    # A100 40GB - 4 GPUs
GPU_TYPE="${GPU_TYPE:-nvidia-h100-80gb}"        # H100 80GB name
if [[ -n "${GPU_COUNT-}" ]]; then
  GPU_COUNT_SOURCE="env"
else
  GPU_COUNT_SOURCE="default"
fi
GPU_COUNT="${GPU_COUNT:-4}"

DEFAULT_SOURCE_INSTANCE=""
mapfile -t _instances_in_zone < <(gcloud compute instances list \
  --project="$PROJECT_ID" \
  --filter="zone:(${DEFAULT_COMPUTE_ZONE})" \
  --format="value(name)" 2>/dev/null)
if [[ ${#_instances_in_zone[@]} -eq 1 ]]; then
  DEFAULT_SOURCE_INSTANCE="${_instances_in_zone[0]}"
fi
mapfile -t _instances_all < <(gcloud compute instances list \
  --project="$PROJECT_ID" \
  --format="value(name,zone)" 2>/dev/null)

# New VM behavior (create from ML image instead of snapshot)
CREATE_FROM_IMAGE="${CREATE_FROM_IMAGE:-false}"  # Set to "true" to create a fresh VM
ML_IMAGE_PROJECT="${ML_IMAGE_PROJECT:-deeplearning-platform-release}"
ML_IMAGE_FAMILY="${ML_IMAGE_FAMILY:-pytorch-2-7-cu128-ubuntu-2404-nvidia-570}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-100}"

# Regions to search (EU first, then Asia, then US)
REGIONS="${REGIONS:-europe-west1 europe-west2 europe-west3 europe-west4 europe-west8 europe-west9 europe-central2 asia-northeast1 asia-northeast2 asia-northeast3 asia-east1 asia-east2 asia-southeast1 asia-southeast2 us-central1 us-east1 us-east4 us-west1 us-west2 us-west4}"

# Optional networking overrides (use if the machine image’s original subnet doesn’t exist in target region)
NETWORK="${NETWORK:-}"        # e.g., "default" or your VPC name
SUBNET="${SUBNET:-}"          # e.g., "subnet-eu-west4"
TAGS="${TAGS:-}"              # e.g., "http-server,https-server"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"  # e.g., "my-sa@${PROJECT_ID}.iam.gserviceaccount.com"
SCOPES="${SCOPES:-}"          # e.g., "https://www.googleapis.com/auth/cloud-platform"

# Snapshot behavior
SNAPSHOT_PREFIX="${SNAPSHOT_PREFIX:-jeff}"  # Prefix for snapshot names (for easy cleanup)
SKIP_SNAPSHOT="${SKIP_SNAPSHOT:-false}"  # Set to "true" to skip creating disk snapshot

# Reservation behavior
SKIP_RESERVATIONS="${SKIP_RESERVATIONS:-true}"  # Set to "false" to use reservations (slower but more reliable)

# Cleanup behavior
DELETE_RESERVATION_AFTER="${DELETE_RESERVATION_AFTER:-true}"

# ====== display current gcloud context ======
format_value() {
  local value="$1"
  if [[ -n "$value" ]]; then
    echo "$value"
  else
    echo "<unset>"
  fi
}

echo "Active gcloud context:"
echo "  Project : $(format_value "$PROJECT_ID")"
echo "  Account : $(format_value "$GCLOUD_ACCOUNT")"
echo "  Zone    : $(format_value "$DEFAULT_COMPUTE_ZONE")"
echo "  Region  : $(format_value "$DEFAULT_COMPUTE_REGION")"
echo

INTERACTIVE_MODE="false"
# ====== interactive mode: choose workflow if none specified ======
if [[ -z "$SOURCE_INSTANCE" && -z "$SOURCE_ZONE" && "${SKIP_SNAPSHOT}" == "false" && "${CREATE_FROM_IMAGE}" == "false" ]]; then
  echo "Select workflow:"
  echo "  1) Migrate an existing VM (create snapshot)"
  echo "  2) Use an existing snapshot"
  echo "  3) Create a new GPU VM from ML image"
  read -r -p "Choose [1-3] (default 1): " _workflow
  _workflow="${_workflow:-1}"
  case "$_workflow" in
    1) ;; # default path
    2) SKIP_SNAPSHOT="true" ;;
    3) CREATE_FROM_IMAGE="true" ;;
    *) echo "Invalid choice: $_workflow"; exit 1 ;;
  esac
  INTERACTIVE_MODE="true"
fi

# ====== interactive prompts for missing required values ======
if [[ "${INTERACTIVE_MODE}" == "true" ]]; then
  GPU_TYPE_OPTIONS=(
    "nvidia-h100-80gb"
    "nvidia-h200-141gb"
    "nvidia-h100-80gb-mega"
    "nvidia-a100-80gb"
    "nvidia-a100-40gb"
    "nvidia-l4"
    "nvidia-tesla-a100"
  )
  GPU_TYPE="$(select_from_list_with_default "Select GPU type:" "$GPU_TYPE" "${GPU_TYPE_OPTIONS[@]}")"

  # Filter machine types based on GPU compatibility
  # H100/H200/B200 → A3 only, A100 → A2 only, L4 → G2 only
  case "$GPU_TYPE" in
    nvidia-h100*|nvidia-h200*|nvidia-b200*)
      MACHINE_TYPE_OPTIONS=(
        "a3-highgpu-8g"
        "a3-highgpu-1g"
        "a3-megagpu-8g"
      )
      MACHINE_TYPE_DEFAULT="a3-highgpu-8g"
      echo "Note: $GPU_TYPE requires A3 machine types." >&2
      ;;
    nvidia-a100*|nvidia-tesla-a100)
      # A100 40GB uses a2-highgpu-*, A100 80GB uses a2-ultragpu-*
      if [[ "$GPU_TYPE" == "nvidia-a100-80gb" ]]; then
        MACHINE_TYPE_OPTIONS=(
          "a2-ultragpu-8g"
          "a2-ultragpu-2g"
          "a2-ultragpu-1g"
        )
        MACHINE_TYPE_DEFAULT="a2-ultragpu-2g"
      else
        # A100 40GB or tesla-a100
        MACHINE_TYPE_OPTIONS=(
          "a2-highgpu-4g"
          "a2-highgpu-2g"
          "a2-highgpu-1g"
        )
        MACHINE_TYPE_DEFAULT="a2-highgpu-4g"
      fi
      echo "Note: $GPU_TYPE requires A2 machine types." >&2
      ;;
    nvidia-l4)
      MACHINE_TYPE_OPTIONS=(
        "g2-standard-8"
        "g2-standard-4"
        "g2-standard-16"
        "g2-standard-32"
      )
      MACHINE_TYPE_DEFAULT="g2-standard-8"
      echo "Note: $GPU_TYPE requires G2 machine types." >&2
      ;;
    *)
      # For other GPUs, show all options
      MACHINE_TYPE_OPTIONS=(
        "a3-highgpu-8g"
        "a3-highgpu-1g"
        "a3-megagpu-8g"
        "a2-highgpu-4g"
        "a2-highgpu-2g"
        "a2-highgpu-1g"
        "a2-ultragpu-8g"
        "a2-ultragpu-2g"
        "a2-ultragpu-1g"
        "g2-standard-8"
        "g2-standard-4"
      )
      MACHINE_TYPE_DEFAULT="$MACHINE_TYPE"
      ;;
  esac
  MACHINE_TYPE="$(select_from_list_with_default "Select machine type:" "$MACHINE_TYPE_DEFAULT" "${MACHINE_TYPE_OPTIONS[@]}")"

  if [[ "$GPU_COUNT_SOURCE" != "env" ]]; then
    case "$MACHINE_TYPE" in
      a3-highgpu-8g|a3-megagpu-8g|a2-ultragpu-8g)
        GPU_COUNT=8
        ;;
      a3-highgpu-1g|a2-highgpu-1g|a2-ultragpu-1g)
        GPU_COUNT=1
        ;;
      a2-highgpu-2g|a2-ultragpu-2g)
        GPU_COUNT=2
        ;;
      a2-highgpu-4g)
        GPU_COUNT=4
        ;;
      g2-standard-4|g2-standard-8)
        GPU_COUNT=1
        ;;
      g2-standard-16)
        GPU_COUNT=2
        ;;
      g2-standard-32)
        GPU_COUNT=4
        ;;
    esac
    echo "GPU count set to $GPU_COUNT for machine type $MACHINE_TYPE."
  fi
fi
if [[ "${CREATE_FROM_IMAGE}" == "true" ]]; then
  # Validate ML image family before scanning regions (but be lenient - let gcloud fail later if needed)
  # Check if the family exists, but don't fail if the check itself has issues
  _family_check_output=$(gcloud compute images list \
    --project="$ML_IMAGE_PROJECT" \
    --filter="family=$ML_IMAGE_FAMILY" \
    --format="value(name)" 2>&1)
  
  if ! echo "$_family_check_output" | grep -qE '^[a-z]'; then
    # Family not found or error occurred - try to find alternatives
    echo "Warning: Could not verify ML image family: $ML_IMAGE_PROJECT/$ML_IMAGE_FAMILY" >&2
    echo "Searching for available PyTorch image families..." >&2
    
    # Try to list PyTorch families
    mapfile -t _ml_families < <(gcloud compute images list \
      --project="$ML_IMAGE_PROJECT" \
      --filter="family~pytorch" \
      --format="value(family)" 2>&1 | grep -v '^ERROR' | sort -u)
    
    # If that fails, try listing all families to see what's available
    if [[ ${#_ml_families[@]} -eq 0 ]]; then
      echo "No PyTorch families found. Listing all available image families..." >&2
      mapfile -t _ml_families < <(gcloud compute images list \
        --project="$ML_IMAGE_PROJECT" \
        --format="value(family)" 2>&1 | grep -v '^ERROR' | sort -u | head -20)
    fi
    
    if [[ ${#_ml_families[@]} -gt 0 ]]; then
      _ml_recommended=()
      for fam in "${_ml_families[@]}"; do
        if pytorch_version_ok "$fam"; then
          _ml_recommended+=("$fam")
        fi
      done
      if [[ -z "$ML_IMAGE_FAMILY" && ${#_ml_recommended[@]} -gt 0 ]]; then
        ML_IMAGE_FAMILY="${_ml_recommended[0]}"
      fi
      if [[ "$ML_IMAGE_FAMILY" != "" && ! " ${_ml_families[*]} " =~ " ${ML_IMAGE_FAMILY} " && ${#_ml_recommended[@]} -gt 0 ]]; then
        ML_IMAGE_FAMILY="${_ml_recommended[0]}"
      fi
      if [[ ${#_ml_recommended[@]} -gt 0 ]]; then
        echo "Recommended (pytorch>=2.1): ${_ml_recommended[*]}" >&2
      fi
      ML_IMAGE_FAMILY="$(select_from_list_with_default "Select ML image family:" "$ML_IMAGE_FAMILY" "${_ml_families[@]}")"
    else
      # If no families found, allow manual entry or proceed with the original
      echo "Could not automatically find image families." >&2
      echo "Will attempt to use: $ML_IMAGE_PROJECT/$ML_IMAGE_FAMILY" >&2
      echo "If this fails, you can set ML_IMAGE_FAMILY manually." >&2
    fi
  fi
  if [[ -z "$NEW_INSTANCE_NAME" ]]; then
    NEW_INSTANCE_NAME="$(prompt_required "Enter the new instance name" "gpu-instance")"
  fi
elif [[ "${SKIP_SNAPSHOT}" == "true" ]]; then
  # If skipping snapshot, ask for snapshot name and region
  if [[ -z "$DISK_SNAPSHOT_NAME" ]]; then
    DISK_SNAPSHOT_NAME="$(prompt_required "Enter the disk snapshot name")"
  fi
  
  if [[ -z "$SNAPSHOT_REGION" ]]; then
    SNAPSHOT_REGION="$(prompt_required "Enter the snapshot region (e.g., us-central1, europe-west1, asia-northeast1)" "$DEFAULT_COMPUTE_REGION")"
  fi
  
  # For new instance name, try to extract from snapshot name or use default
  if [[ -z "$NEW_INSTANCE_NAME" ]]; then
    # Try to extract instance name from snapshot name (remove prefix and timestamp)
    EXTRACTED_NAME=$(echo "$DISK_SNAPSHOT_NAME" | sed "s/^${SNAPSHOT_PREFIX}-snapshot-//" | sed 's/-[0-9]\{8\}-[0-9]\{6\}$//')
    if [[ -n "$EXTRACTED_NAME" ]]; then
      NEW_INSTANCE_NAME="${EXTRACTED_NAME}-gpu"
    else
      NEW_INSTANCE_NAME="gpu-instance"
    fi
  fi
else
  # If creating snapshot, ask for source instance details
  if [[ -z "$SOURCE_INSTANCE" ]]; then
    if [[ -n "$DEFAULT_SOURCE_INSTANCE" ]]; then
      SOURCE_INSTANCE="$(prompt_required "Enter the source instance name" "$DEFAULT_SOURCE_INSTANCE")"
    elif [[ ${#_instances_in_zone[@]} -gt 1 ]]; then
      SOURCE_INSTANCE="$(select_from_list "Select the source instance in $DEFAULT_COMPUTE_ZONE:" "${_instances_in_zone[@]}")"
    else
      SOURCE_INSTANCE="$(select_from_list "Select the source instance (format: name or name zone):" "${_instances_all[@]}")"
    fi
  fi

  if [[ -z "$SOURCE_ZONE" ]]; then
    SOURCE_ZONE="$(prompt_required "Enter the source instance zone" "$DEFAULT_COMPUTE_ZONE")"
  fi

  # Generate names based on source instance (after prompts or env vars are set)
  DISK_SNAPSHOT_NAME="${DISK_SNAPSHOT_NAME:-${SNAPSHOT_PREFIX}-snapshot-$(echo "$SOURCE_INSTANCE" | tr '_' '-')-$(date +%Y%m%d-%H%M%S)}"
  NEW_INSTANCE_NAME="${NEW_INSTANCE_NAME:-${SOURCE_INSTANCE}-gpu}"
fi

# ====== sanity checks ======
if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: Missing PROJECT_ID."
  exit 1
fi

if [[ "${CREATE_FROM_IMAGE}" == "true" ]]; then
  if [[ -z "$NEW_INSTANCE_NAME" ]]; then
    echo "Error: Missing NEW_INSTANCE_NAME when CREATE_FROM_IMAGE=true."
    echo "Usage: CREATE_FROM_IMAGE=true NEW_INSTANCE_NAME=<name> [MACHINE_TYPE=...] [GPU_COUNT=...] [BOOT_DISK_SIZE_GB=100] [REGIONS=\"...\"] $0"
    echo "Example: CREATE_FROM_IMAGE=true NEW_INSTANCE_NAME=my-gpu-vm ./launch_vm.sh"
    exit 1
  fi
elif [[ "${SKIP_SNAPSHOT}" == "true" ]]; then
  if [[ -z "$DISK_SNAPSHOT_NAME" || -z "$SNAPSHOT_REGION" ]]; then
    echo "Error: Missing DISK_SNAPSHOT_NAME or SNAPSHOT_REGION when SKIP_SNAPSHOT=true."
    echo "Usage: SKIP_SNAPSHOT=true DISK_SNAPSHOT_NAME=<snapshot> SNAPSHOT_REGION=<region> [NEW_INSTANCE_NAME=...] [MACHINE_TYPE=...] [GPU_COUNT=...] [REGIONS=\"...\"] $0"
    echo "Example: SKIP_SNAPSHOT=true DISK_SNAPSHOT_NAME=my-snapshot SNAPSHOT_REGION=us-central1 ./launch_vm.sh"
    exit 1
  fi
else
  if [[ -z "$SOURCE_INSTANCE" || -z "$SOURCE_ZONE" ]]; then
    echo "Error: Missing required values. Please provide SOURCE_INSTANCE and SOURCE_ZONE."
    echo "Usage: SOURCE_INSTANCE=<name> SOURCE_ZONE=<zone> [NEW_INSTANCE_NAME=...] [MACHINE_TYPE=...] [GPU_COUNT=...] [SNAPSHOT_PREFIX=...] [SKIP_SNAPSHOT=true] [REGIONS=\"...\"] $0"
    echo "Example: SOURCE_INSTANCE=my-training-vm SOURCE_ZONE=us-east1-b ./launch_vm.sh"
    exit 1
  fi
fi

echo "Project         : $PROJECT_ID"
if [[ "${CREATE_FROM_IMAGE}" == "true" ]]; then
  echo "Image family    : $ML_IMAGE_FAMILY ($ML_IMAGE_PROJECT)"
  echo "Boot disk size  : ${BOOT_DISK_SIZE_GB}GB"
elif [[ "${SKIP_SNAPSHOT}" == "true" ]]; then
  echo "Disk snapshot   : $DISK_SNAPSHOT_NAME ($SNAPSHOT_REGION)"
else
  echo "Source instance : $SOURCE_INSTANCE ($SOURCE_ZONE)"
  echo "Disk snapshot   : $DISK_SNAPSHOT_NAME"
fi
echo "New instance    : $NEW_INSTANCE_NAME"
echo "GPU             : $GPU_COUNT x $GPU_TYPE on $MACHINE_TYPE"
echo

# ====== 1) Create disk snapshot from source instance (optional) ======
if [[ "${CREATE_FROM_IMAGE}" == "true" ]]; then
  echo ">> Skipping disk snapshot creation (CREATE_FROM_IMAGE=true)"
elif [[ "${SKIP_SNAPSHOT}" != "true" ]]; then
  echo ">> Creating disk snapshot \"$DISK_SNAPSHOT_NAME\" from $SOURCE_INSTANCE ..."
  # Get the boot disk name from the source instance
  BOOT_DISK_NAME=$(gcloud compute instances describe "$SOURCE_INSTANCE" \
    --project="$PROJECT_ID" \
    --zone="$SOURCE_ZONE" \
    --format="value(disks[0].source)" | sed 's|.*/||')
  
  gcloud compute disks snapshot "$BOOT_DISK_NAME" \
    --project="$PROJECT_ID" \
    --zone="$SOURCE_ZONE" \
    --snapshot-names="$DISK_SNAPSHOT_NAME" \
    >/dev/null

  echo "✓ Disk snapshot created: $DISK_SNAPSHOT_NAME"
else
  echo ">> Skipping disk snapshot creation (SKIP_SNAPSHOT=true)"
  echo ">> Using existing disk snapshot: $DISK_SNAPSHOT_NAME"
fi
echo

# ====== helper: list zones in a region ======
list_zones() {
  local region="$1"
  gcloud compute zones list \
    --project="$PROJECT_ID" \
    --filter="region=$region" \
    --format="value(name)"
}

# ====== helper: map display GPU name to actual GCP accelerator type ======
get_gcp_gpu_type() {
  local display_name="$1"
  case "$display_name" in
    nvidia-a100-40gb)
      # A100 40GB is called nvidia-tesla-a100 in GCP
      echo "nvidia-tesla-a100"
      ;;
    *)
      # For all other types, use the display name as-is
      echo "$display_name"
      ;;
  esac
}

# ====== helper: check if zone offers the specified GPU ======
zone_offers_gpu() {
  local zone="$1"
  local gcp_gpu_type
  gcp_gpu_type="$(get_gcp_gpu_type "$GPU_TYPE")"
  local available_types
  available_types=$(gcloud compute accelerator-types list \
    --project="$PROJECT_ID" \
    --filter="zone:$zone" \
    --format="value(name)" 2>/dev/null)
  
  if [[ -z "$available_types" ]]; then
    echo "     (no accelerators available in $zone)"
    return 1
  fi
  
  if echo "$available_types" | grep -q -E "^${gcp_gpu_type}$"; then
    return 0
  else
    echo "     (available: $(echo "$available_types" | tr '\n' ' '))"
    return 1
  fi
}

# ====== helper: check if zone supports machine type ======
zone_supports_machine_type() {
  local zone="$1"
  local machine_type="$2"
  
  gcloud compute machine-types list \
    --project="$PROJECT_ID" \
    --zones="$zone" \
    --filter="name=$machine_type" \
    --format="value(name)" 2>/dev/null | grep -q "^${machine_type}$"
}

# ====== helper: try to reserve capacity in a zone ======
probe_zone_capacity() {
  local zone="$1"
  local res="res-gpu-${zone}-$(date +%s)"
  local gcp_gpu_type
  gcp_gpu_type="$(get_gcp_gpu_type "$GPU_TYPE")"

  set +e
  gcloud compute reservations create "$res" \
    --project="$PROJECT_ID" \
    --zone="$zone" \
    --vm-count=1 \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="count=${GPU_COUNT},type=${gcp_gpu_type}" \
    --require-specific-reservation \
    >/dev/null 2>&1
  local rc=$?
  set -e

  if (( rc != 0 )); then
    echo "  ❌ Reservation failed in $zone (no capacity or quota)." >&2
    return 1
  fi

  echo "  ✅ Reservation held in $zone: $res" >&2
  echo "$res"
  return 0
}

# ====== 2) Build candidate zones (that OFFER the specified GPU) in preferred regions ======
CANDIDATE_ZONES=()
echo ">> Scanning regions for zones that OFFER $GPU_TYPE ..."
for r in $REGIONS; do
  echo " Region $r:"
  zones_in_region=$(list_zones "$r")
  if [[ -z "$zones_in_region" ]]; then
    echo "   - No zones found in region $r"
    continue
  fi
  for z in $zones_in_region; do
    echo "   - Checking zone $z..."
    if zone_offers_gpu "$z" && zone_supports_machine_type "$z" "$MACHINE_TYPE"; then
      echo "   - $z ($GPU_TYPE + $MACHINE_TYPE offered)"
      CANDIDATE_ZONES+=("$z")
    else
      if ! zone_offers_gpu "$z"; then
        echo "   - $z (no $GPU_TYPE)"
      elif ! zone_supports_machine_type "$z" "$MACHINE_TYPE"; then
        echo "   - $z ($GPU_TYPE offered but no $MACHINE_TYPE)"
      fi
    fi
  done
done

if [[ ${#CANDIDATE_ZONES[@]} -eq 0 ]]; then
  echo "No zones found that offer $GPU_TYPE in the given regions."
  exit 2
fi
echo

# ====== 3) Probe capacity and create instance from machine image ======
for ZONE in "${CANDIDATE_ZONES[@]}"; do
  if [[ "${SKIP_RESERVATIONS}" == "true" ]]; then
    echo ">> Skipping reservations, trying direct creation in $ZONE ..."
    RES_ID=""
  else
    echo ">> Probing capacity in $ZONE ..."
    RES_ID="$(probe_zone_capacity "$ZONE" || echo "")"
    if [[ -z "$RES_ID" ]]; then
      continue
    fi
  fi

  # Create unique instance name per zone to avoid conflicts
  # Convert zone to valid instance name format (lowercase, hyphens only)
  ZONE_SUFFIX=$(echo "$ZONE" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  UNIQUE_INSTANCE_NAME="${NEW_INSTANCE_NAME}-${ZONE_SUFFIX}"
  
  # Ensure name is not too long (Google Cloud limit is 63 characters)
  if [[ ${#UNIQUE_INSTANCE_NAME} -gt 63 ]]; then
    # Truncate and add zone suffix
    MAX_BASE_LENGTH=$((63 - ${#ZONE_SUFFIX} - 1))  # -1 for the hyphen
    TRUNCATED_BASE=$(echo "$NEW_INSTANCE_NAME" | cut -c1-$MAX_BASE_LENGTH)
    UNIQUE_INSTANCE_NAME="${TRUNCATED_BASE}-${ZONE_SUFFIX}"
  fi
  
  # Check if instance already exists
  if gcloud compute instances describe "$UNIQUE_INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
    echo "  ⚠️  Instance $UNIQUE_INSTANCE_NAME already exists in $ZONE, skipping..."
    continue
  fi
  
  if [[ "${CREATE_FROM_IMAGE}" == "true" ]]; then
    echo ">> Creating instance \"$UNIQUE_INSTANCE_NAME\" in $ZONE from image family \"$ML_IMAGE_FAMILY\" ..."
  else
    echo ">> Creating instance \"$UNIQUE_INSTANCE_NAME\" in $ZONE from disk snapshot \"$DISK_SNAPSHOT_NAME\" ..."
  fi
  
  # Get snapshot size to ensure disk is large enough
  if [[ "${CREATE_FROM_IMAGE}" != "true" ]]; then
    if [[ "${SKIP_SNAPSHOT}" == "true" ]]; then
      SNAPSHOT_SIZE=$(gcloud compute snapshots describe "$DISK_SNAPSHOT_NAME" \
        --project="$PROJECT_ID" \
        --region="$SNAPSHOT_REGION" \
        --format="value(diskSizeGb)" 2>/dev/null || echo "100")
    else
      SNAPSHOT_SIZE=$(gcloud compute snapshots describe "$DISK_SNAPSHOT_NAME" \
        --project="$PROJECT_ID" \
        --format="value(diskSizeGb)" 2>/dev/null || echo "100")
    fi
  fi
  
  # Map display GPU name to actual GCP accelerator type
  GCP_GPU_TYPE="$(get_gcp_gpu_type "$GPU_TYPE")"
  
  # Build common flags for instance creation
  CREATE_FLAGS=( --project="$PROJECT_ID"
                 --zone="$ZONE"
                 --machine-type="$MACHINE_TYPE"
                 --accelerator="count=${GPU_COUNT},type=${GCP_GPU_TYPE}"
                 --maintenance-policy=TERMINATE
                 --no-preemptible
                 --boot-disk-type=pd-ssd
               )

  if [[ "${CREATE_FROM_IMAGE}" == "true" ]]; then
    CREATE_FLAGS+=( --image-family="$ML_IMAGE_FAMILY"
                    --image-project="$ML_IMAGE_PROJECT"
                    --boot-disk-size="${BOOT_DISK_SIZE_GB}GB"
                  )
  else
    CREATE_FLAGS+=( --source-snapshot="$DISK_SNAPSHOT_NAME"
                    --boot-disk-size="${SNAPSHOT_SIZE}GB"
                  )
  fi

  # Optional overrides that may be needed across regions
  [[ -n "$NETWORK" ]] && CREATE_FLAGS+=( --network="$NETWORK" )
  [[ -n "$SUBNET" ]] && CREATE_FLAGS+=( --subnet="$SUBNET" )
  [[ -n "$TAGS" ]] && CREATE_FLAGS+=( --tags="$TAGS" )
  [[ -n "$SERVICE_ACCOUNT" ]] && CREATE_FLAGS+=( --service-account="$SERVICE_ACCOUNT" )
  [[ -n "$SCOPES" ]] && CREATE_FLAGS+=( --scopes="$SCOPES" )

  # Attach to the specific reservation so we actually use the capacity we just held.
  if [[ -n "$RES_ID" ]]; then
    CREATE_FLAGS+=( --reservation-affinity=specific --reservation="$RES_ID" )
  fi

  # Create the instance
  echo "Attempting to create instance (this may take a few minutes)..."
  set +e
  gcloud compute instances create "$UNIQUE_INSTANCE_NAME" "${CREATE_FLAGS[@]}" 2>&1 | tee /tmp/gcloud_create_${ZONE}.log
  CREATE_RC=$?
  set -e
  
  if [[ $CREATE_RC -eq 0 ]]; then
    echo "✓ Instance created: $UNIQUE_INSTANCE_NAME ($ZONE)"
    # Optional cleanup: delete reservation (instance is now running using its own capacity)
    if [[ "${DELETE_RESERVATION_AFTER}" == "true" ]]; then
      echo ">> Deleting reservation $RES_ID ..."
      gcloud compute reservations delete "$RES_ID" --project="$PROJECT_ID" --zone="$ZONE" --quiet >/dev/null || true
    else
      echo "Keeping reservation $RES_ID (you can delete it later)."
    fi
    echo
    echo "All set. SSH: gcloud compute ssh $UNIQUE_INSTANCE_NAME --zone $ZONE"
    exit 0
  else
    # Check if instance was created despite gcloud error
    echo "  ⚠️  gcloud command failed, checking if instance was created anyway..."
    sleep 5  # Give it time to show up
    if gcloud compute instances describe "$UNIQUE_INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
      echo "  ✅ Instance $UNIQUE_INSTANCE_NAME was created successfully despite gcloud error!"
      echo "  Instance may still be provisioning. Check status with:"
      echo "     gcloud compute instances describe $UNIQUE_INSTANCE_NAME --zone $ZONE"
      echo "  SSH: gcloud compute ssh $UNIQUE_INSTANCE_NAME --zone $ZONE"
      exit 0
    fi
    
    echo "  ❌ Instance creation failed in $ZONE. Cleaning reservation and trying next zone..."
    # Only try to delete reservation if we have a valid reservation ID
    if [[ -n "$RES_ID" && "$RES_ID" =~ ^[a-z][-a-z0-9]*$ ]]; then
      gcloud compute reservations delete "$RES_ID" --project="$PROJECT_ID" --zone="$ZONE" --quiet >/dev/null || true
    fi
    # Add small delay to avoid rate limiting
    sleep 2
  fi
done

echo "No capacity found that could create the instance. Try expanding REGIONS or running later."
exit 3