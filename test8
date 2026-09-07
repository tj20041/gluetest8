import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, sum as spark_sum, count, when
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sessionization_pipeline")

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Enable Spark ANSI SQL strict arithmetic compliance
spark.conf.set("spark.sql.ansi.enabled", "true")

logger.info("Reading user clickstream session logs...")

clickstream_data = [
    ("US", "campaign_spring", 1, 120),
    ("US", "campaign_spring", 2, 45),
    ("EU", "campaign_spring", 0, 0),     # Zero activity cohort
    ("EU", "campaign_winter", 1, 300),
    ("APAC", "campaign_direct", 0, 0)   # Zero activity cohort
]

schema = ["region", "campaign", "conversions", "bounce_count"]
click_df = spark.createDataFrame(clickstream_data, schema)

logger.info("Aggregating marketing conversions by regional cohort...")

agg_cohorts = click_df.groupBy("region", "campaign").agg(
    spark_sum("conversions").alias("total_conversions"),
    spark_sum("bounce_count").alias("total_bounces"),
    count("*").alias("cohort_size")
)

logger.info("Computing conversion-to-bounce index...")

# FAILS HERE: When total_bounces is 0, ANSI compliance raises SparkArithmeticException: Division by zero
conversion_metric_df = agg_cohorts.withColumn(
    "efficiency_ratio",
    col("total_conversions") / col("total_bounces")
)

conversion_metric_df.show()
job.commit()
