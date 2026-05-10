import aioboto3
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO

from core.config import settings

class S3Client:
    
    def __init__(
        self,
        host: str = settings.S3_HOST,
        port: int = settings.S3_PORT,
        access_key: str = settings.S3_ACCESS_KEY,
        secret_key: str = settings.S3_SECRET_KEY,
        bucket_name: str = settings.S3_BUCKET_NAME,
        secure: bool = settings.S3_SECURE,
    ):
        self.endpoint_url = f"http://{host}:{port}"        
        self.aws_access_key_id = access_key
        self.aws_secret_access_key = secret_key
        self.bucket_name = bucket_name
        self.secure = secure
        self.session = aioboto3.Session()
        self.client = None
        
    async def start(self):
        self.client = await self.session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            use_ssl=self.secure,
        ).__aenter__()

    async def stop(self):
        if self.client:
            await self.client.__aexit__(None, None, None)

    async def create_bucket_if_not_exists(self):
        try:
            await self.client.create_bucket(Bucket=self.bucket_name)
            print(f"Бакет '{self.bucket_name}' успешно создан.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                print(f"Бакет '{self.bucket_name}' уже существует.")
            else:
                print(f"Ошибка при создании бакета: {e}")

    async def upload_fileobj(self, file_obj: BinaryIO, object_name: str) -> Optional[str]:
        try:
            await self.client.upload_fileobj(file_obj, self.bucket_name, object_name)
            return object_name
        except ClientError as e:
            print(f"Ошибка загрузки файла в S3: {e}")
            return None
