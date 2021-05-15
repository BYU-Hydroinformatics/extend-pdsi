#!/bin/sh
# requires curl
cat urls.txt | tr -d '\r' | xargs -n 1 -P 4 curl -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies
mkdir original-data
mv GLDAS*.nc4 original-data/