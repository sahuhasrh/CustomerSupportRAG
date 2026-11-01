
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

LOCAL_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
CLOUD_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
COLLECTION = os.getenv(
    "QDRANT_CLOUD_COLLECTION",
    "customer_support_faq_1000"
)

if not CLOUD_URL or not CLOUD_API_KEY:
    raise ValueError("Qdrant Cloud credentials are missing from .env")

local = QdrantClient(url=LOCAL_URL)
cloud = QdrantClient(url=CLOUD_URL, api_key=CLOUD_API_KEY)

# Read local collection configuration
info = local.get_collection(COLLECTION)
expected_count = info.points_count

print(f"Local collection: {COLLECTION}")
print(f"Points to migrate: {expected_count}")

if expected_count == 0:
    raise RuntimeError("Local collection is empty.")

# Avoid accidentally overwriting an existing cloud collection
if cloud.collection_exists(COLLECTION):
    raise RuntimeError(
        f"Cloud collection '{COLLECTION}' already exists. "
        "Migration stopped to avoid overwriting data."
    )

# Create cloud collection with the same vector configuration
cloud.create_collection(
    collection_name=COLLECTION,
    vectors_config=info.config.params.vectors,
    on_disk_payload=True
)

# Copy vectors and payloads in batches
offset = None
migrated = 0
batch_size = 100

while True:
    points, next_offset = local.scroll(
        collection_name=COLLECTION,
        limit=batch_size,
        offset=offset,
        with_payload=True,
        with_vectors=True
    )

    if not points:
        break

    cloud.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=p.id,
                vector=p.vector,
                payload=p.payload
            )
            for p in points
        ],
        wait=True
    )

    migrated += len(points)
    print(f"Migrated {migrated}/{expected_count}")

    if next_offset is None:
        break

    offset = next_offset

# Verify cloud collection
cloud_count = cloud.count(
    collection_name=COLLECTION,
    exact=True
).count

print("\nMigration verification")
print(f"Expected points: {expected_count}")
print(f"Cloud points: {cloud_count}")

if cloud_count != expected_count:
    raise RuntimeError("Migration count mismatch!")

print("\nSUCCESS: Qdrant migration completed.")

local.close()
cloud.close()
