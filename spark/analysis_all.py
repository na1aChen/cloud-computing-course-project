"""
A-1 + A-2 合并脚本：数据清洗 + SQL统计分析
输出写入/tmp/out.txt，容器执行后cat读取
"""
out = open("/tmp/out.txt", "w")
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, round as _round, row_number,
    desc, split, explode, isnan
)
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("MovieAnalysis").getOrCreate()

# ========== A-1 ==========
out.write("=" * 60 + "\n")
out.write("A-1 数据清洗\n")
out.write("=" * 60 + "\n")

df = spark.read.option("header", True).option("inferSchema", True).option("encoding", "UTF-8").option("multiLine", True).csv("/opt/spark/work/douban_movies.csv")

out.write("=== Schema ===\n")
out.write(df._jdf.schema().treeString() + "\n")

out.write("\n=== 前5行 ===\n")
for row in df.select("movie_id", "title", "year", "rating_score", "rating_count", "genres").take(5):
    out.write(str(row) + "\n")

total = df.count()
out.write(f"\n总行数: {total}\n")
out.write(f"{'字段':<20} {'缺失数':>10} {'缺失比例':>10}\n")
out.write("-" * 42 + "\n")
num_types = ("int", "bigint", "double", "float")
for c in df.columns:
    col_type = dict(df.dtypes).get(c, "")
    if col_type in num_types:
        null_count = df.filter(col(c).isNull() | isnan(col(c))).count()
    else:
        null_count = df.filter(col(c).isNull()).count()
    out.write(f"{c:<20} {null_count:>10} {null_count/total:>9.2%}\n")

before = df.count()
df = df.dropna(subset=["rating_score"])
year_mode = df.groupBy("year").count().orderBy(col("count").desc()).first()
if year_mode:
    df = df.fillna({"year": year_mode["year"]})
df = df.fillna({"genres": "未知"})
after = df.count()
out.write(f"\n清洗前行数: {before}，清洗后: {after}，删除: {before - after}\n")

out.write("\n=== 清洗后统计 ===\n")
num_cols = [f for f, t in df.dtypes if t in ("int", "bigint", "double", "float")]
if num_cols:
    desc = df.select(num_cols).describe().collect()
    for r in desc:
        out.write(str(r) + "\n")

# ========== A-2 ==========
out.write("\n" + "=" * 60 + "\n")
out.write("A-2 Spark SQL 统计分析\n")
out.write("=" * 60 + "\n")

# Q1
out.write("\n=== Q1: 各类型电影数量和平均评分 ===\n")
q1 = df.withColumn("genre_list", explode(split(col("genres"), "/")))\
  .groupBy("genre_list")\
  .agg(count("*").alias("movie_count"), _round(avg("rating_score"), 2).alias("avg_rating"))\
  .filter(col("genre_list") != "未知")\
  .orderBy(desc("movie_count"))
for r in q1.take(15):
    out.write(str(r) + "\n")

# Q2
out.write(f"\n=== Q2: 评分最高Top-10(评分人数>10000) ===\n")
q2 = df.filter(col("rating_count") > 10000)\
  .orderBy(desc("rating_score"))\
  .select("title", "rating_score", "year", "rating_count", "genres")\
  .limit(10)
for r in q2.collect():
    out.write(str(r) + "\n")

# Q3
out.write("\n=== Q3: 年度电影评分趋势(1990年起) ===\n")
q3 = df.filter(col("year").isNotNull())\
  .filter(col("year") >= 1990)\
  .groupBy("year")\
  .agg(count("*").alias("movie_count"), _round(avg("rating_score"), 2).alias("avg_rating"))\
  .orderBy("year")
for r in q3.take(15):
    out.write(str(r) + "\n")

# Q4
out.write("\n=== Q4: 各类型内评分排名(窗口函数, Top-3) ===\n")
df_exploded = df.withColumn("genre_single", explode(split(col("genres"), "/")))
window_spec = Window.partitionBy("genre_single").orderBy(desc("rating_score"))
q4 = df_exploded.filter(col("rating_count") > 5000)\
  .withColumn("rank_in_genre", row_number().over(window_spec))\
  .filter(col("rank_in_genre") <= 3)\
  .select("genre_single", "title", "rating_score", "year", "rank_in_genre")\
  .orderBy("genre_single", "rank_in_genre")
for r in q4.take(30):
    out.write(str(r) + "\n")

out.write("\n=== A-1 + A-2 全部完成 ===\n")
out.flush()
out.close()
spark.stop()
