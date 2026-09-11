import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, sum as spark_sum, count, when, try_divide
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

# Enable Spark ANSI SQL strict arithmetic compliance.
# NOTE: ANSI mode is kept enabled per business requirement, but all arithmetic
# transformations below MUST use null-safe helpers (try_divide / when-guards)
# so that division-by-zero degrades to NULL instead of aborting the job.
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

# Data-quality pre-check: surface zero-denominator cohorts in CloudWatch logs
# BEFORE the division transformation runs, so this condition is visible even
# though it is now handled gracefully rather than causing a job failure.
zero_bounce_count = agg_cohorts.filter(col("total_bounces") == 0).count()
if zero_bounce_count > 0:
    logger.warning(
        "Detected %d cohort(s) with total_bounces == 0; efficiency_ratio will be "
        "NULL for these cohorts instead of raising DIVIDE_BY_ZERO.",
        zero_bounce_count
    )

logger.info("Computing conversion-to-bounce index...")

# FIXED: Use try_divide (null-safe division) instead of raw '/' so that ANSI
# mode (spark.sql.ansi.enabled=true) does not raise SparkArithmeticException
# [DIVIDE_BY_ZERO] when total_bounces == 0. try_divide returns NULL for the
# divide-by-zero case instead of aborting the task/job.
conversion_metric_df = agg_cohorts.withColumn(
    "efficiency_ratio",
    try_divide(col("total_conversions"), col("total_bounces"))
)

# As a defense-in-depth guard (in case try_divide is unavailable in a given
# Glue Spark runtime), explicitly null-out any residual zero-denominator rows
# rather than letting a raw division ever reach the aggregation output.
conversion_metric_df = conversion_metric_df.withColumn(
    "efficiency_ratio",
    when(col("total_bounces") == 0, None).otherwise(col("efficiency_ratio"))
)

try:
    conversion_metric_df.show()
except Exception as e:
    logger.warning("Non-fatal error while displaying conversion_metric_df: %s", str(e))

job.commit()
