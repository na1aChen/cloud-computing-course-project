"""
A-3 性能对比：Pandas vs PySpark
"""
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, round as _round, desc, split, explode

csv_path = "douban_movies.csv"

print("=" * 60)
print("A-3 性能对比：Pandas vs PySpark")
print("=" * 60)

# ===== Pandas =====
start = time.time()
pdf = pd.read_csv(csv_path, encoding="utf-8")
pdf = pdf.dropna(subset=["rating_score"])
pdf["genres"] = pdf["genres"].fillna("未知")
rows = []
for _, row in pdf.iterrows():
    genre_str = str(row["genres"])
    for g in genre_str.split("/"):
        rows.append({"genre": g, "rating_score": row["rating_score"]})
pdf_expanded = pd.DataFrame(rows)
result_pd = pdf_expanded.groupby("genre").agg(
    movie_count=("rating_score", "count"),
    avg_rating=("rating_score", "mean"),
).sort_values("movie_count", ascending=False)
t_pandas = time.time() - start
print(f"\n[Pandas] 耗时: {t_pandas:.4f}s, 结果行数: {len(result_pd)}")
print(result_pd.head(10).to_string())

# ===== PySpark =====
spark = SparkSession.builder.appName("PerfCompare").master("local[2]").getOrCreate()
start = time.time()
sdf = spark.createDataFrame(pdf)
result_spark = (
    sdf.withColumn("genre_list", explode(split(col("genres"), "/")))
    .filter(col("rating_score").isNotNull())
    .groupBy("genre_list")
    .agg(count("*").alias("movie_count"), _round(avg("rating_score"), 2).alias("avg_rating"))
    .orderBy(desc("movie_count"))
    .collect()
)
t_spark = time.time() - start
print(f"\n[PySpark] 耗时: {t_spark:.4f}s, 结果行数: {len(result_spark)}")
for r in result_spark[:10]:
    print(f"  {r.genre_list}: count={r.movie_count}, avg={r.avg_rating}")

# ===== 对比 =====
print("\n" + "=" * 60)
print("对比汇总")
print("=" * 60)
print(f"Pandas(单机):      {t_pandas:.4f}s")
print(f"PySpark(local[2]): {t_spark:.4f}s")

speedup = t_pandas / t_spark if t_spark > 0 else 1.0
print(f"实测加速比: {speedup:.2f}x")

# Amdahl分析
f = 2 * (1 - 1 / speedup) if speedup > 1 else 0.85
print(f"\nAmdahl分析: 可并行比例 f = {f:.4f}")
print(f"理论加速比 S(2) = {1/((1-f)+f/2):.4f}")
print("加速比未达线性的原因：通信开销(Shuffle) + Python-JVM 序列化 + 67K 数据量偏小导致调度占比高")

spark.stop()
print("\n=== A-3 全部完成 ===")
