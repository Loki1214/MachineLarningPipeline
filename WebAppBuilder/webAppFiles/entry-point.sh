#!/bin/sh


until curl "http://storage:9001" >/dev/null 2>&1; do
  echo "waiting for the storage to start..."
  sleep 2
done
echo "Succeeded in connecting to the storage server."

until mysqladmin ping -h database --silent; do
  echo "waiting for the database to start..."
  sleep 2
done
echo "Succeeded in connecting to the database."

exec "$@"