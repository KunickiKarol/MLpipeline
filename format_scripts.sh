directory="$1"

if [[ -z "$directory" ]]; then
  echo "Usage: ./format_scripts.sh <directory>"
  exit 1
fi

black "$directory"
isort --profile black scripts
