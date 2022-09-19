#!/bin/sh


until curl "http://${STORAGE}:9001" >/dev/null 2>&1; do
  echo "waiting for ${STORAGE} to start..."
  sleep 2
done
echo "Succeeded in connecting to ${STORAGE} server."

until mysqladmin ping -h mysql --silent; do
  echo "waiting for mysqld to start..."
  sleep 2
done
echo "Succeeded in connecting to mysqld."

exec "$@"