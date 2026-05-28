"""
A-2 Spark SQL统计分析(15分)
4个查询：GROUP BY、Top-N、时间趋势、窗口函数
使用清洗后的数据，转为Spark DataFrame进行分析
"""
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, round as _round, row_number,
    desc, split, explode
)
from pyspark.sql.window import Window


def load(spark, path):
    pdf = pd.read_csv(path, encoding="utf-8")
    return spark.createDataFrame(pdf)


def query1_group_by_genre(df):
    """GROUP BY聚合：按类型统计电影数量和平均评分"""
    result = (
        df.withColumn("genre_list", explode(split(col("genres"), "/")))
        .groupBy("genre_list")
        .agg(
            count("*").alias("movie_count"),
            _round(avg("rating_score"), 2).alias("avg_rating"),
        )
        .filter(col("genre_list") != "未知")
        .orderBy(desc("movie_count"))
    )
    print("\n=== Q1: 各类型电影数量和平均评分 ===")
    result.show(15, truncate=False)
    return result


def query2_top_n(df, n=10):
    """ORDER BY Top-N：评分最高的N部电影(过滤评分人数>10000)"""
    result = (
        df.filter(col("rating_count") > 10000)
        .orderBy(desc("rating_score"))
        .select("title", "rating_score", "year", "rating_count", "genres")
        .limit(n)
    )
    print(f"\n=== Q2: 评分最高Top-{n}(评分人数>10000) ===")
    result.show(n, truncate=False)
    return result


def query3_year_trend(df):
    """时间维度趋势分析：每年电影数量和平均评分变化"""
    result = (
        df.filter(col("year").isNotNull())
        .filter(col("year") >= 1990)
        .groupBy("year")
        .agg(
            count("*").alias("movie_count"),
            _round(avg("rating_score"), 2).alias("avg_rating"),
        )
        .orderBy("year")
    )
    print("\n=== Q3: 年度电影评分趋势(1990年起) ===")
    result.show(15, truncate=False)
    return result


def query4_window_rank(df):
    """窗口函数：各类型内按评分排名"""
    df_exploded = df.withColumn("genre_single", explode(split(col("genres"), "/")))
    window_spec = Window.partitionBy("genre_single").orderBy(desc("rating_score"))
    result = (
        df_exploded.filter(col("rating_count") > 5000)
        .withColumn("rank_in_genre", row_number().over(window_spec))
        .filter(col("rank_in_genre") <= 3)
        .select("genre_single", "title", "rating_score", "year", "rank_in_genre")
        .orderBy("genre_single", "rank_in_genre")
    )
    print("\n=== Q4: 各类型内评分排名(窗口函数，Top-3) ===")
    result.show(30, truncate=False)
    return result


def main():
    spark = SparkSession.builder.appName("MovieAnalysis").getOrCreate()
    df = load(spark, "douban_movies_clean.csv")

    query1_group_by_genre(df)
    query2_top_n(df)
    query3_year_trend(df)
    query4_window_rank(df)

    spark.stop()


if __name__ == "__main__":
    main()
