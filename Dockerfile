FROM yytest_sentence_bench:v1

WORKDIR /usr/local/bin/

COPY main_v7.py /usr/local/bin/
#COPY benchmark_rerank.py /usr/local/bin/

# 暴露端口
EXPOSE 35000

# 启动命令
CMD ["gunicorn", \
    "--bind", "0.0.0.0:35000", \
    "--workers", "1", \
    "--timeout", "300", \
    "main_v7:app"]
