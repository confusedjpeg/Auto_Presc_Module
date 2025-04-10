import boto3
from botocore.exceptions import ClientError
from app.core.config import settings
import logging
from typing import Optional, BinaryIO
import uuid

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.S3_BUCKET_NAME

    async def upload_file(self, file: BinaryIO, file_name: str, content_type: str) -> Optional[str]:
        """
        Upload a file to S3 bucket
        """
        try:
            # Generate a unique file name to avoid collisions
            unique_filename = f"{uuid.uuid4()}_{file_name}"
            
            self.s3_client.upload_fileobj(
                file,
                self.bucket_name,
                unique_filename,
                ExtraArgs={'ContentType': content_type}
            )
            
            # Generate the URL for the uploaded file
            url = f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{unique_filename}"
            return url
            
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            return None

    async def get_file_url(self, file_name: str) -> Optional[str]:
        """
        Generate a presigned URL for file access
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': file_name
                },
                ExpiresIn=3600  # URL expires in 1 hour
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            return None

    async def delete_file(self, file_name: str) -> bool:
        """
        Delete a file from S3 bucket
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_name
            )
            return True
        except ClientError as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            return False

s3_service = S3Service() 