#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
minio_image="minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
mc_image="quay.io/minio/mc@sha256:a7fe349ef4bd8521fb8497f55c6042871b2ae640607cf99d9bede5e9bdf11727"
container_name="daft-lance-native-reader-minio-$$"
network_name="daft-lance-native-reader-$$"
bucket_name="daft-lance-native-reader"
access_key="daftlance"
secret_key="daftlance-secret"

cleanup() {
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
  docker network rm "${network_name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker network create "${network_name}" >/dev/null
docker run --detach --name "${container_name}" --network "${network_name}" --network-alias minio \
  --publish 127.0.0.1::9000 \
  --env "MINIO_ROOT_USER=${access_key}" \
  --env "MINIO_ROOT_PASSWORD=${secret_key}" \
  "${minio_image}" server /data >/dev/null

host_port="$(docker port "${container_name}" 9000/tcp | awk -F: 'NR == 1 {print $NF}')"
endpoint="http://127.0.0.1:${host_port}"

for _ in $(seq 1 60); do
  if curl --fail --silent "${endpoint}/minio/health/ready" >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent "${endpoint}/minio/health/ready" >/dev/null

docker run --rm --network "${network_name}" --entrypoint /bin/sh "${mc_image}" -c \
  "mc alias set local http://minio:9000 '${access_key}' '${secret_key}' >/dev/null && mc mb --ignore-existing local/${bucket_name} >/dev/null"

export DAFT_LANCE_RUN_FAST_PATH_INTEGRATION=1
export DAFT_LANCE_TEST_S3_URI="s3://${bucket_name}/datasets"
export DAFT_LANCE_TEST_S3_ENDPOINT="${endpoint}"
export DAFT_LANCE_TEST_S3_ACCESS_KEY="${access_key}"
export DAFT_LANCE_TEST_S3_SECRET_KEY="${secret_key}"
export DAFT_LANCE_TEST_S3_REGION="us-east-1"

cd "${repo_root}"
uv run --with 'ray[data,client]==2.55.1' python -m pytest -q tests/io/lancedb/test_fast_path_merge_integration.py
