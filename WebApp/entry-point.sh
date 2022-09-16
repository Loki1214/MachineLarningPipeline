#!/bin/sh


until curl 'http://minio:9001'; do
  echo 'waiting for minio to start...'
  sleep 2
done
echo 'Succeeded in connecting to minio server.'

until mysqladmin ping -h mysql --silent; do
  echo 'waiting for mysqld to start...'
  sleep 2
done
echo 'Succeeded in connecting to mysqld.'

exec "$@"