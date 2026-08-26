"""Typed, stable errors for the offline diarization contracts."""

class SDDiarError(Exception):
    """Base class for contract/runtime failures."""

class ContractValidationError(SDDiarError, ValueError):
    """An object violates an explicit data contract."""

class TimebaseInvariantViolation(ContractValidationError):
    code = "TIMEBASE_INVARIANT_VIOLATION"

class ResultSchemaValidationFailed(ContractValidationError):
    code = "RESULT_SCHEMA_VALIDATION_FAILED"

class ProtectedOverlapError(ContractValidationError):
    code = "PROTECTED_OVERLAP_VIOLATION"


class ModelPackError(SDDiarError):
    """A signed offline model-pack cannot be used safely."""


class ModelNotFound(ModelPackError):
    code = "MODEL_NOT_FOUND"


class ModelHashMismatch(ModelPackError):
    code = "MODEL_HASH_MISMATCH"


class ModelRuntimeIncompatible(ModelPackError):
    code = "MODEL_RUNTIME_INCOMPATIBLE"


class ManifestSignatureInvalid(ModelPackError):
    code = "MANIFEST_SIGNATURE_INVALID"


class OfflinePolicyViolation(SDDiarError):
    code = "OFFLINE_POLICY_VIOLATION"
