import boto3
import os
import logging
from PIL import Image
from io import BytesIO

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # Set up AWS S3 client
    s3 = boto3.client('s3')

    # Specify the thumbnail dimensions
    THUMBNAIL_SIZE = (128, 128)

    # Loop through all records in the event (handles multiple uploads)
    for record in event['Records']:
        # Get bucket and object key from the event
        source_bucket = record['s3']['bucket']['name']
        source_key = record['s3']['object']['key']

        # Specify destination key in the "output" folder
        destination_key = f"output/{source_key}"

        logger.info(f"Processing file: {source_key} from bucket: {source_bucket}")

        try:
            # Download the image from S3
            response = s3.get_object(Bucket=source_bucket, Key=source_key)
            logger.info(f"Downloaded file: {source_key} from bucket: {source_bucket}")
            image_data = response['Body'].read()

            # Open the image and resize it
            try:
                with Image.open(BytesIO(image_data)) as img:
                    img.verify()  # Verify it's a valid image
                    logger.info(f"Verified image: {source_key}")

                    img.thumbnail(THUMBNAIL_SIZE)
                    buffer = BytesIO()
                    img.save(buffer, img.format)
                    buffer.seek(0)

                    # Upload the thumbnail back to S3 into the "output" folder
                    s3.put_object(Bucket=source_bucket, Key=destination_key, Body=buffer, ContentType=response['ContentType'])
                    logger.info(f"Thumbnail created and uploaded to {source_bucket}/{destination_key}")
            except Exception as img_error:
                logger.error(f"Error processing image {source_key}: {img_error}")
                continue

        except Exception as e:
            logger.error(f"Error accessing file {source_bucket}/{source_key}: {str(e)}")
            raise e

    return {
        'statusCode': 200,
        'body': 'Thumbnails created successfully.'
    }
