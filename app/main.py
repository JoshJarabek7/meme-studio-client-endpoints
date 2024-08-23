"""Main entrance file for the FastAPI application."""

import asyncio
import datetime
from uuid import uuid4

import aiohttp
import runpod
from azure.storage.blob import BlobSasPermissions, ContainerSasPermissions, generate_blob_sas, generate_container_sas
from fastapi import FastAPI
from loguru import logger
from pydantic import AnyUrl, BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

"---------- CONSTANTS, VARIABLES, CONFIGURATIONS, & SETTINGS ----------"


class AzureStorageSettings(BaseSettings):
    """Contains the settings class for Azure Storage environment variables."""

    model_config = SettingsConfigDict(env_prefix="azure_storage__")
    name: str = ""  # AZURE_STORAGE__NAME
    key: str = ""  # AZURE_STORAGE__KEY
    input_container: str = ""  # AZURE_STORAGE__INPUT_CONTAINER
    output_container: str = ""  # AZURE_STORAGE__OUTPUT_CONTAINER


class RunpodSettings(BaseSettings):
    """Contains the settings class for RunPod environment variables."""

    model_config = SettingsConfigDict(env_prefix="runpod__")
    endpoint_id: str = ""  # RUNPOD__ENDPOINT_ID
    key: str = ""  # RUNPOD__KEY


az = AzureStorageSettings()
rp = RunpodSettings()

asu = f"https://{az.name}.blob.core.windows.net/"  # Azure Storage URL
runpod.api_key = rp.key


"---------- SCHEMAS ----------"


class RequestInput(BaseModel):
    """Sent by the client to begin the segmentation process."""

    points: list[tuple[int | float, int | float]] | None = None
    labels: list[int] | None = None
    bbox: list[list[int | float]] | None = None
    textual: str | None = None
    automatic: bool = False
    input_url: AnyUrl

    @model_validator(mode="after")
    def validate_groups(self) -> "RequestInput":
        """Validates the client request to ensure the prompting is correct."""
        group_one_active = self.points is not None or self.labels is not None or self.bbox is not None
        group_two_active = self.textual is not None
        group_three_active = self.automatic is True

        total_active = group_one_active + group_two_active + group_three_active

        if total_active < 1:
            raise ValueError(
                """
                --- No segmentation method provided ---
                Please specify points and labels, bounding box, textual prompt, or set automatic to True.
                """,
            )
        if total_active > 1:
            raise ValueError(
                """
                --- Multiple segmentation methods detected ---
                Please use only one method:
                \t1. points, labels, bounding box
                \t2. textual prompt
                \t3. automatic
                """,
            )

        if self.bbox and len(self.bbox) != 4:  # noqa: PLR2004
            raise ValueError(
                """
                    --- Invalid bounding box format ---
                    Expected [x_min, y_min, x_max, y_max].
                    """,
            )

        if self.points or self.labels:
            if bool(self.points) != bool(self.labels):
                raise ValueError(
                    """
                    Both points and labels must be provided together for point-based segmentation.
                    """,
                )
            if self.points and self.labels and len(self.points) != len(self.labels):
                msg = f"""
                    --- Mismatch between points and labels ---
                    Got {len(self.points)} points and {len(self.labels)} labels.
                    """
                raise ValueError(
                    msg,
                )

        if self.labels and not all(label in [0, 1] for label in self.labels):
            raise ValueError(
                """
                    --- Invalid label values ---
                    Labels must only contain 0 (negative) or 1 (positive).
                    """,
            )

        return self


class RunpodInputPayload(RequestInput):
    """Inherits RequestInput. Passed to RunPod instance as the payload for inference."""

    output_sas: str  # SAS token to append to URL for creating/writing blobs
    output_prefix: str = f"{asu}{az.output_container}/BLOBHERE?SASHERE"


class BeginSegmentationSchema(BaseModel):
    """Entry-point request to generate segmentations for the client."""

    url: str


class SegmentationResponseModel(BaseModel):
    """Reponse Model for a segmentation run."""

    urls: list[AnyUrl]
    status: str


"---------- UTILITY FUNCTIONS ----------"


def generate_full_url(container_name: str, blob_name: str, token: str) -> AnyUrl:
    """Combine the container name, blob name, and sas token to create a full URL."""
    return f"{asu}{container_name}/{blob_name}?{token}"


def get_client_write_sas() -> str:
    """Generate the write sas for the client to upload their image to a blob."""
    blob_name = str(uuid4())
    return generate_full_url(
        container_name=az.input_container,
        blob_name=str(uuid4()),
        token=generate_blob_sas(
            account_name=az.name,
            container_name=az.input_container,
            blob_name=blob_name,
            account_key=az.key,
            permission=BlobSasPermissions(write=True),
            expiry=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
            start=datetime.datetime.now(datetime.UTC),
        ),
    )


def get_runpod_write_sas() -> str:
    """Generate the container sas for the runpod container to upload the results to."""
    return generate_container_sas(
        account_name=az.name,
        container_name=az.output_container,
        account_key=az.key,
        permission=ContainerSasPermissions(write=True),
        expiry=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        start=datetime.datetime.now(datetime.UTC),
    )


def get_output_read_sas(blob_name: str) -> str:
    """Generate read token for the client to download their processed image."""
    return generate_full_url(
        container_name=az.output_container,
        blob_name=blob_name,
        token=generate_blob_sas(
            account_name=az.name,
            container_name=az.output_container,
            blob_name=blob_name,
            account_key=az.key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
            start=datetime.datetime.now(datetime.UTC),
        ),
    )


"---------- ROUTES ----------"

app = FastAPI()


@app.get("/api/sas")
async def get_client_write_sas_route() -> dict[str, str]:
    """Route to generate the write token for the client to upload their image to a blob."""
    blob_name, sas_token = get_client_write_sas()
    return {"url": generate_full_url(container_name=az.input_container, blob_name=blob_name, token=sas_token)}


@app.post("/api/segment", response_model=SegmentationResponseModel)
async def create_segmentations_route(begin_seg: RequestInput) -> SegmentationResponseModel:
    """Generate the segmentations on the uploaded image."""
    input_payload = RunpodInputPayload(
        points=begin_seg.points,
        labels=begin_seg.labels,
        bbox=begin_seg.bbox,
        textual=begin_seg.textual,
        automatic=begin_seg.automatic,
        input_url=begin_seg.input_url,
        output_sas=get_runpod_write_sas(),
    ).model_dump()
    async with aiohttp.ClientSession() as session:
        endpoint = runpod.AsyncioEndpoint(rp.endpoint_id, session)
        job: runpod.AsyncioJob = await endpoint.run(input_payload)
        round_counter = 0
        round_seconds = 0.0
        logger.info(f"Round: {round_counter + 1}. Total seconds ran: {round_seconds}.")
        # Polling job status
        while True:
            round_counter += 1
            # Get current status of job
            status = await job.status()
            logger.info(f"Current job status: {status}")

            # Successful job run
            if status == "COMPLETED":
                job_output = await job.output()
                blob_names = job_output["blob_names"]
                return SegmentationResponseModel.model_validate(
                    {"urls": [get_output_read_sas(blob_name) for blob_name in blob_names], "status": "COMPLETED"},
                )

            # Failed job run
            if status in ["FAILED"]:
                logger.error("Job failed or encountered an error.")
                return SegmentationResponseModel.model_validate({"urls": [], "status": "FAILED"})

            round_counter += 1
            round_seconds += 2.5

            # In-Progress job run
            logger.info(f"Job ID: {job.job_id} --- in queue or processing. Waiting 2.5 seconds...")
            await asyncio.sleep(2.5)
