from sqlalchemy import (
    MetaData, Table, Column, Integer, String, DateTime, ForeignKey, Date
)
from enum import StrEnum

ExternalSystemName = "DITM"

# TODO Move all column-name strings to StrEnum 

class EmailCols(StrEnum):
    """ A consistent, single-source for column names.

    These are used in the Table declarations below,
    when specifying columns to access the database in Loader,
    and when validating the data in Preprocessor (note:
    CSV validation must exclude database-exclusive columns).

    Only include columns needed for processing
    """
    contentid       = "ContentId"
    emailsubject    = "EmailSubject"
    bodytext        = "BodyText"
    timestamp       = "ReceivedDateUtc"
    # toaddresses     = "ToAddresses"
    # fromaddress     = "FromAddress"

class TranscriptCols(StrEnum):
    contentid        = "ContentId"
    # transcriptid     = "TranscriptId"
    timestamp        = "TranscriptDate"
    # externalsystemid = "ExternalSystemId"
    # companyid        = "CompanyId"
    contentbody      = "ContentBody"

#####
class ExternalSystemsCols(StrEnum):
    externalsystemid = "ExternalSystemId"
    systemname       = "SystemName"

class ContentCols(StrEnum):
    contentid = "ContentId"

class ContentExportCols(StrEnum):
    contentid          = "ContentId"
    externalsystemid   = "ExternalSystemId"
    exportstatus       = "ExportStatus"
    exportdateutc      = "ExportDateUtc"
    lastattemptdateutc = "LastAttemptDateUtc"
    externalidentifier = "ExternalIdentifier"
    failurereason      = "FailureReason"

metadata = MetaData()

content = Table(
    "Content",
    metadata,
    Column(ContentCols.contentid, Integer, primary_key = True)
)

contentexports = Table(
    "ContentExports",
    metadata,
    Column(
        ContentExportCols.contentid,
        Integer,
        ForeignKey("Content.ContentId"),
        nullable = False
    ),
    Column(
        ContentExportCols.externalsystemid,
        Integer,
        ForeignKey("ExternalSystems.ExternalSystemId"),
        nullable = False
    ),
    Column(ContentExportCols.exportstatus, String(30), nullable = False),
    Column(ContentExportCols.exportdateutc, DateTime, nullable = True),
    Column(ContentExportCols.externalidentifier, String(500), nullable = True),
    Column(ContentExportCols.lastattemptdateutc, DateTime, nullable = True),
    Column(ContentExportCols.failurereason, String(4000), nullable = True),
)


emailmessages = Table(
    "EmailMessages",
    metadata,
    Column(
        "ContentId",
        Integer,
        ForeignKey('Content.ContentId'),
        primary_key = True,
        nullable = False,
    ),
    Column("MessageId", String(500), nullable = True),
    Column("GraphItemId", String(300), nullable = True),
    Column("SourceMailbox", String(320), nullable = True),
    Column("FromAddress", String(500), nullable = True),
    Column("SenderName", String(500), nullable = True),
    Column("ToAddresses", String(None), nullable = True),
    Column("EmailSubject", String(1000), nullable = True),
    Column("SentDateUtc", DateTime, nullable = True),
    Column("ReceivedDateUtc", DateTime, nullable = True),
    Column("BodyText", String(None), nullable = True),
    Column("BodyHtml", String(None), nullable = True),
)


transcripts = Table(
    "Transcripts",
    metadata,
    Column(
        "ContentId",
        Integer,
        ForeignKey('Content.ContentId'),
        primary_key = True,
        nullable = False,
    ),
    Column("TranscriptId", Integer, nullable = False),
    Column("CompanyId", Integer, nullable = False),
    Column("TranscriptDate", DateTime, nullable = False),
    Column(
        "ExternalSystemId",
        Integer,
        ForeignKey("ExternalSystems.ExternalSystemId"),
        nullable = False
    ),
    Column("ContentBody", String(None), nullable = False),
    Column("DateAddedUtc", Date, nullable = False),
)



externalsystems = Table(
    "ExternalSystems",
    metadata,
    Column("ExternalSystemId", Integer, primary_key = True),
    Column("SystemName", String(100), nullable = False, unique = True),
    Column("Description", String(500), nullable = True),
)
