# Resumable dataset downloads shared by the GPU runners. Source it with DATA_ROOT set.
# cold_get <lab> <part> <seq_cond>: COLD recording into $DATA_ROOT/seq_data (images, odometry, places).
# cid_get <file_id> <zip> <md5> <target>: CID-SIMS sequence into $DATA_ROOT/<target>, MD5-checked.
# fetch_all: one cold_get/cid_get argument line per stdin line, DL_PARALLEL (4) at a time.

cold_get() {  # lab part seq_cond
  local lab=$1 part=$2 name=$3
  local target="$DATA_ROOT/seq_data/cold-${lab}_part_${part}_${name}"
  if [ -d "$target/odom_scans" ] && [ -d "$target/std_cam" ]; then echo "$target: present"; return; fi
  local tarball="$DATA_ROOT/seq_data/_${lab}_${part}_${name}.tar"
  for attempt in 1 2 3 4 5; do
    curl -sS -L -C - --retry 5 -o "$tarball" "https://www.cas.kth.se/COLD/db/cold-${lab}/part_${part}/${name}.tar" && break
    sleep 15
  done
  rm -rf "$target.partial" && mkdir -p "$target.partial"
  local top=$(tar -tf "$tarball" 2>/dev/null | head -1 | cut -d/ -f1)
  tar -xf "$tarball" -C "$target.partial" --strip-components=1 --wildcards \
      "${top}/std_cam/*" "${top}/odom_scans/*" "${top}/localization/*" \
    || { echo "$tarball: EXTRACTION FAILED (delete it to re-download)"; return 1; }
  mv "$target.partial" "$target" && rm -f "$tarball"
  echo "$target: OK ($(ls "$target/std_cam" | wc -l) images)"
}
cid_get() {  # file_id zip md5 target
  local id=$1 zip=$2 md5=$3 target="$DATA_ROOT/$4"
  if [ -f "$target/odom.txt" ] && [ -d "$target/color" ]; then echo "$target: present"; return; fi
  local archive="$DATA_ROOT/$zip"
  for attempt in 1 2 3 4 5; do
    curl -sS -L -C - --retry 5 -o "$archive" \
      "https://download.scidb.cn/download?fileId=$id&dataSetType=organization&fileName=$zip" && break
    sleep 15
  done
  [ "$(md5sum "$archive" | cut -d' ' -f1)" = "$md5" ] || { echo "$archive: MD5 MISMATCH (delete it to re-download)"; return 1; }
  local staging="$DATA_ROOT/_staging_$4"
  rm -rf "$staging" && mkdir -p "$staging"
  unzip -q -o "$archive" '*color/*' '*odom.txt' '*groundtruth.txt' -d "$staging" || { echo "$archive: UNZIP FAILED"; return 1; }
  if [ -d "$staging/color" ]; then mv "$staging" "$target"
  else mv "$staging/$(ls "$staging" | head -1)" "$target" && rm -rf "$staging"; fi
  rm -f "$archive"
  echo "$target: OK ($(ls "$target/color" | wc -l) images)"
}
export -f cold_get cid_get

fetch_all() {
  # xargs -L joins a line ending in a blank with the next one, so trailing blanks are removed.
  sed 's/[[:space:]]*$//' | xargs -P "${DL_PARALLEL:-4}" -L 1 bash -c \
    'kind=$1; shift; if [ "$kind" = cold ]; then cold_get "$@"; else cid_get "$@"; fi' _
}
