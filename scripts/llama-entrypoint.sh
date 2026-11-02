#!/bin/sh
# Starts llama-server with the arguments `make use` wrote to /state/llama.args (one per line).
set -eu
if [ ! -s /state/llama.args ]; then
  echo "no model selected: run 'make use MODEL=<id>' on the host" >&2
  exit 1
fi
set --
while IFS= read -r arg || [ -n "$arg" ]; do
  [ -n "$arg" ] && set -- "$@" "$arg"
done < /state/llama.args
exec /app/llama-server "$@"
