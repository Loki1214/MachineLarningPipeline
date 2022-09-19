#!/bin/sh

echo "CREATE DATABASE ${MYSQL_DATABASE};" | mysql -u${MYSQL_USER} -p${MYSQL_PASSWORD} --port=3306