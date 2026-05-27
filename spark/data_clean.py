"""
A-1 数据清洗（10分）
数据集：豆瓣电影评分（约200MB）
使用 Spark DataFrame API 完成数据清洗
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, stddev, min as _min, max as _max, when, isnan


def load_data(spark, data_path):
    """加载数据，支持 CSV / Parquet / s3a 路径"""
    if data_path.endswith(".csv") or data_path.endswith(".csv.gz"):
        df = spark.read.option("header", True).option("inferSchema", True).csv(data_path)
    elif data_path.endswith(".parquet"):
        df = spark.read.parquet(data_path)
    else:
        df = spark.read.option("header", True).option("inferSchema", True).csv(data_path)
    return df


def analyze_missing(df):
    """统计各字段缺失值比例"""
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
    """对至少2个有缺失值的字段采用不同处理策略"""
    before_count = df.count()

    # 策略1：对评分缺失的行直接删除（dropna）—— 评分是核心字段，缺失无法推断
    df = df.dropna(subset=["rating"])

    # 策略2：对年份缺失的值用众数填充（fillna）—— 年份可合理推断
    year_mode = df.groupBy("year").count().orderBy(col("count").desc()).first()
    if year_mode:
        mode_year = year_mode["year"]
        df = df.fillna({"year": mode_year})

    # 策略3（可选）：对类型缺失的填充为"未知"
    if "genre" in df.columns:
        df = df.fillna({"genre": "未知"})

    after_count = df.count()
    print(f"\n清洗前行数: {before_count}，清洗后行数: {after_count}，删除: {before_count - after_count}")
    return df


def basic_stats(df):
    """输出各字段基本统计信息"""
    num_cols = [f for f, t in df.dtypes if t in ("int", "bigint", "double", "float")]
    if num_cols:
        df.select(num_cols).describe().show()


def main():
    spark = SparkSession.builder.appName("MovieDataClean").getOrCreate()

    # 本地测试用样例数据，实际运行时替换为 s3a://<BUCKET>/douban_movies.csv
    data_path = "s3a://<BUCKET>/douban_movies.csv"

    df = load_data(spark, data_path)

    print("=== Schema ===")
    df.printSchema()

    print("\n=== 前5行 ===")
    df.show(5, truncate=False)

    analyze_missing(df)

    df_clean = clean_data(df)

    print("\n=== 清洗后统计 ===")
    basic_stats(df_clean)

    spark.stop()


if __name__ == "__main__":
    main()
