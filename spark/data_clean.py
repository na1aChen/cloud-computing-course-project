"""
A-1 数据清洗(10分)
数据集：豆瓣电影评分(67,132条，11字段)
使用pandas加载CSV(处理多行摘要)→转为Spark DataFrame完成清洗
"""
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, stddev, min as _min, max as _max, when, isnan


def load_data(spark, data_path):
    pdf = pd.read_csv(data_path, encoding="utf-8")
    df = spark.createDataFrame(pdf)
    return df


def analyze_missing(df):
    total = df.count()
    print(f"\n总行数: {total}")
    print(f"\n{'字段':<20} {'缺失数':>10} {'缺失比例':>10}")
    print("-" * 42)
    num_types = ("int", "bigint", "double", "float")
    for c in df.columns:
        col_type = dict(df.dtypes).get(c, "")
        if col_type in num_types:
            null_count = df.filter(col(c).isNull() | isnan(col(c))).count()
        else:
            null_count = df.filter(col(c).isNull()).count()
        print(f"{c:<20} {null_count:>10} {null_count/total:>9.2%}")


def clean_data(df):
    before_count = df.count()

    # 策略1：评分缺失的行直接删除(dropna)——评分是核心字段，缺失无法推断
    df = df.dropna(subset=["rating_score"])

    # 策略2：年份缺失用众数填充(fillna)——年份可合理推断
    year_mode = df.groupBy("year").count().orderBy(col("count").desc()).first()
    if year_mode:
        df = df.fillna({"year": year_mode["year"]})

    # 策略3：类型缺失的填充为"未知"
    df = df.fillna({"genres": "未知"})

    after_count = df.count()
    print(f"\n清洗前行数: {before_count}，清洗后行数: {after_count}，删除: {before_count - after_count}")
    return df


def basic_stats(df):
    num_cols = [f for f, t in df.dtypes if t in ("int", "bigint", "double", "float")]
    if num_cols:
        df.select(num_cols).describe().show()


def main():
    spark = SparkSession.builder.appName("MovieDataClean").getOrCreate()

    data_path = "douban_movies.csv"

    df = load_data(spark, data_path)

    print("=== Schema ===")
    df.printSchema()

    print("\n=== 前5行 ===")
    df.select("movie_id", "title", "year", "rating_score", "rating_count", "genres").show(5, truncate=False)

    analyze_missing(df)

    df_clean = clean_data(df)

    print("\n=== 清洗后统计 ===")
    basic_stats(df_clean)

    # 保存清洗后的数据
    df_clean.toPandas().to_csv("douban_movies_clean.csv", index=False, encoding="utf-8")
    print("\n清洗后数据已保存: douban_movies_clean.csv")

    spark.stop()


if __name__ == "__main__":
    main()
