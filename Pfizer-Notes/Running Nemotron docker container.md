
```
git clone https://huggingface.co/nvidia/nemotron-ocr-v2
cd nemotron-ocr-v2
docker build -t nemotron-ocr-v2:local .

```

```
docker run --gpus all -it --rm \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  nemotron-ocr-v2:local python example.py --image path/to/your/image.jpg

```