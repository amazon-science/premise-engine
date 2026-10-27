# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Shared utilities: Bedrock client, prompts, metrics, sample loaders."""

from premise.providers import (
    SUPPORTED_PROVIDERS,
    ProviderConfig,
    ProviderConfigurationError,
    ProviderRequestError,
    get_provider_client,
    load_provider_config,
    model_for_role,
    parse_model_specs,
    provider_name,
)
from premise.utils.bedrock import (
    DEFAULT_BATCH_ROLE_ARN,
    DEFAULT_REGION,
    DEFAULT_S3_BUCKET,
    compute_file_hash,
    download_batch_results,
    extract_batch_model_response,
    get_batch_model_input,
    get_bedrock_client,
    get_bedrock_management_client,
    get_s3_client,
    invoke_model,
    refresh_credentials,
    require_aws_setting,
    resolve_batch_settings,
    submit_batch_job,
    upload_to_s3,
    wait_for_batch_job,
)
from premise.utils.metrics import compute_krippendorff_alpha, parse_score
from premise.utils.prompts import PAIRWISE_JUDGE_PROMPT
from premise.utils.samples import load_samples, load_traits

__all__ = [
    "DEFAULT_BATCH_ROLE_ARN",
    "DEFAULT_REGION",
    "DEFAULT_S3_BUCKET",
    "PAIRWISE_JUDGE_PROMPT",
    "SUPPORTED_PROVIDERS",
    "ProviderConfig",
    "ProviderConfigurationError",
    "ProviderRequestError",
    "compute_file_hash",
    "compute_krippendorff_alpha",
    "download_batch_results",
    "extract_batch_model_response",
    "get_batch_model_input",
    "get_bedrock_client",
    "get_bedrock_management_client",
    "get_provider_client",
    "get_s3_client",
    "invoke_model",
    "load_provider_config",
    "load_samples",
    "load_traits",
    "model_for_role",
    "parse_model_specs",
    "parse_score",
    "provider_name",
    "refresh_credentials",
    "require_aws_setting",
    "resolve_batch_settings",
    "submit_batch_job",
    "upload_to_s3",
    "wait_for_batch_job",
]
