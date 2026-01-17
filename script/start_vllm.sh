CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve \
       --model /data/yangchenyu/pre-trained-models/Qwen3-30B-A3B \
       --served-model-name Qwen3-30B-A3B \
       --max-model-len 32000 \
       --enable-expert-parallel \
       --tensor-parallel-size 4 \
       --enable-auto-tool-choice \
       --tool-call-parser qwen3_coder \
       --port 8000

# CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#        --model /hpc2hdd/home/bli303/hf_models/Qwen3-Coder-30B-A3B-Instruct \
#        --served-model-name Qwen3-Coder-30B-A3B-Instruct \
#        --max-model-len 32000 \
#        --enable-expert-parallel \
#        --tensor-parallel-size 2 \
#        --enable-auto-tool-choice \
#        --tool-call-parser qwen3_coder \
#        --port 8000