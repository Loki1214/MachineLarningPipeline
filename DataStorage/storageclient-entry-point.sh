#!/bin/sh

until curl 'http://minio:9001' >/dev/null 2>&1; do
  echo 'waiting for minio to start...'
  sleep 2
done
echo 'Succeeded in connecting to minio server.'

mc alias set minio http://minio:9000 minioadminuser minioadminpassword --api S3v4

exec "$@"