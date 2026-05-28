from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("WordCount").getOrCreate()

# 从CSV的summary字段读取文本做词频统计
lines = spark.read.option("header", True).option("encoding", "UTF-8").csv("/opt/spark/work/douban_movies.csv")

word_counts = (
    lines.select("summary")
    .rdd.flatMap(lambda row: str(row[0]).split())
    .map(lambda word: (word, 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: x[1], ascending=False)
)

import sys
result = str(word_counts.take(10))
with open("/tmp/result.txt", "w") as f:
    f.write("Top 10 words: " + result + "\n")
    f.write("WordCount job completed successfully!\n")
print("Top 10 words:", result, flush=True)
print("RESULT_FILE=/tmp/result.txt", flush=True)
spark.stop()
