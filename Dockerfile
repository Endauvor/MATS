FROM akkadeeemikk/qwen2_5

RUN pip install unsloth
RUN pip install --no-deps --upgrade "flash-attn>=2.6.3"
RUN pip install jupyter plotly
