echo "Run this script 'source env.sh' to set the env variables for Call Me Maybe"
read -p "intra: " INTRA 2>/dev/null || read "INTRA?intra: "

echo Setting UV_CACHE_DIR to /goinfre/$INTRA/uv_cache
export UV_CACHE_DIR=/goinfre/$INTRA/uv_cache
if [[ "$UV_CACHE_DIR" ==  "/goinfre/$INTRA/uv_cache" ]]; then
    echo "\tOK"
else
    echo "\tFAIL"
fi
echo Setting UV_PROJECT_ENVIRONMENT to /goinfre/$INTRA/call_me_maybe-venv
export UV_PROJECT_ENVIRONMENT=/goinfre/$INTRA/call_me_maybe-venv
if [[ "$UV_PROJECT_ENVIRONMENT" ==  "/goinfre/$INTRA/call_me_maybe-venv" ]]; then
    echo "\tOK"
else
    echo "\tFAIL"
fi
echo Setting HF_HOME to /goinfre/$INTRA/huggingface-cache
export HF_HOME=/goinfre/$INTRA/huggingface-cache
if [[ "$HF_HOME" ==  "/goinfre/$INTRA/huggingface-cache" ]]; then
    echo "\tOK"
else
    echo "\tFAIL"
fi