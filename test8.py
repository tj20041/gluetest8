import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, sum as spark_sum, count, when, lit
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

# ANSI strict mode is intentionally enabled.
# All division expressions MUST guard against zero denominators explicitly.
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

# Data quality check: log a warning when zero-bounce rows are present so that
# zero-denominator conditions are visible before aggregation.
zero_bounce_count = click_df.filter(col("bounce_count") == 0).count()
if zero_bounce_count > 0:
    logger.warning(
        f"{zero_bounce_count} row(s) with bounce_count=0 detected "
        "— efficiency_ratio will be NULL for these cohorts."
    )

logger.info("Aggregating marketing conversions by regional cohort...")

agg_cohorts = click_df.groupBy("region", "campaign").agg(
    spark_sum("conversions").alias("total_conversions"),
    spark_sum("bounce_count").alias("total_bounces"),
    count("*").alias("cohort_size")
)

logger.info("Computing conversion-to-bounce index...")

# Guard against division by zero: when total_bounces is 0, return NULL instead
# of raising a SparkArithmeticException under ANSI strict arithmetic mode.
conversion_metric_df = agg_cohorts.withColumn(
    "efficiency_ratio",
    when(col("total_bounces") == 0, lit(None))
    .otherwise(col("total_conversions") / col("total_bounces"))
)

try:
    conversion_metric_df.show()
except Exception as e:
    logger.error(f"Failed to display conversion_metric_df: {e}")
    raise

job.commit()
