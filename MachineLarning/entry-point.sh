#!/bin/sh

until curl "http://builder:5000" >/dev/null 2>&1; do
  echo "waiting for the WebAppBuilder to start..."
  sleep 2
done
echo "Succeeded in connecting to the WebAppBuilder."

until curl "http://${STORAGE}:9001" >/dev/null 2>&1; do
  echo "waiting for ${STORAGE} to start..."
  sleep 2
done
echo "Succeeded in connecting to ${STORAGE} server."

until mysqladmin ping -h mysql --silent; do
  echo 'waiting for mysqld to start...'
  sleep 2
done
echo 'Succeeded in connecting to mysqld.'

echo "Executes the command: \"$@\""
exec "$@"