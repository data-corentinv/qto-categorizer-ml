"""Read/Write datasets from/to external sources/destinations."""

# %% IMPORTS

import abc
import typing as T

import awswrangler as wr
import pandas as pd
import pydantic as pdt
import typing_extensions as TX

# %% TYPES

ParquetEngine = T.Literal["auto", "pyarrow", "fastparquet"]
GlueWritingMode = T.Literal["append", "overwrite", "overwrite_partitions"]

# %% READERS


class Reader(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a dataset reader.

    Use a reader to load a dataset in memory.
    e.g., to read file, database, cloud storage, ...
    """

    KIND: str

    @abc.abstractmethod
    def read(self) -> pd.DataFrame:
        """Read a dataframe from a dataset.

        Returns:
            pd.DataFrame: dataframe representation.
        """


class ParquetReader(Reader):
    """Read a dataframe from a parquet file.

    Parameters:
        path (str): local or S3 path to the dataset.
        engine (ParquetEngine): parquet engine to use.
    """

    KIND: T.Literal["ParquetReader"] = "ParquetReader"

    path: str
    engine: ParquetEngine = "fastparquet"

    @TX.override
    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path, engine=self.engine)


class GlueReader(Reader):
    """Read a dataframe from a glue table.

    Parameters:
        table (str): name of the table.
        database (str): name of the database.
        catalog_id (str, optional): data catalog ID.
    """

    KIND: T.Literal["GlueReader"] = "GlueReader"

    table: str
    database: str
    catalog_id: str = "123456789"

    @TX.override
    def read(self) -> pd.DataFrame:
        return wr.s3.read_parquet_table(
            table=self.table, database=self.database, catalog_id=self.catalog_id
        )


class DeltalakeReader(Reader):
    """Read a dataframe from the delta lake.

    Parameters:
        path (str): S3 path to the delta lake file.
    """

    KIND: T.Literal["DeltalakeReader"] = "DeltalakeReader"

    path: str

    @TX.override
    def read(self) -> pd.DataFrame:
        return wr.s3.read_deltalake(path=self.path)


class CSVReader(Reader):
    """Read a dataframe from a csv file.

    Parameters:
        path (str): local path to the dataset.
    """

    KIND: T.Literal["CSVReader"] = "CSVReader"

    path: str
    sep: str = ","
    dtypes: dict = {}
    parse_dates: list = []

    @TX.override
    def read(self) -> pd.DataFrame:
        data = pd.read_csv(self.path, sep=self.sep, dtype=self.dtypes, parse_dates=self.parse_dates)
        return data


ReaderKind = ParquetReader | GlueReader | DeltalakeReader | CSVReader

# %% WRITERS


class Writer(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a dataset writer.

    Use a writer to save a dataset from memory.
    e.g., to write file, database, cloud storage, ...
    """

    KIND: str

    @abc.abstractmethod
    def write(self, data: pd.DataFrame) -> None:
        """Write a dataframe to a dataset.

        Args:
            data (pd.DataFrame): dataframe representation.
        """


class ParquetWriter(Writer):
    """Writer a dataframe to a parquet file.

    Parameters:
        path (str): local or S3 path to the dataset.
        engine (ParquetEngine): parquet engine to use.
    """

    KIND: T.Literal["ParquetWriter"] = "ParquetWriter"

    path: str
    engine: ParquetEngine = "fastparquet"

    @TX.override
    def write(self, data: pd.DataFrame) -> None:
        pd.DataFrame.to_parquet(data, self.path, engine=self.engine)


class GlueWriter(Writer):
    """Write a dataframe to a glue table.

    Parameters:
        table (str): name of the table.
        database (str): name of the database.
        catalog_id (str, optional): data catalog ID.
        mode (GlueWritingMode): write mode for the table.
        path (str | None): associated S3 path (only for creation).
        glue_table_settings (dict[str, T.Any] | None): settings for the table.
    """

    KIND: T.Literal["GlueWriter"] = "GlueWriter"

    table: str
    database: str
    catalog_id: str = "123456789"
    mode: GlueWritingMode = "overwrite"
    path: str | None = None
    glue_table_settings: dict[str, T.Any] | None = None

    @TX.override
    def write(self, data: pd.DataFrame) -> None:
        settings = (
            T.cast(wr.typing.GlueTableSettings, self.glue_table_settings)
            if self.glue_table_settings is not None
            else None
        )
        wr.s3.to_parquet(
            df=data,
            dataset=True,
            mode=self.mode,
            path=self.path,
            table=self.table,
            database=self.database,
            catalog_id=self.catalog_id,
            glue_table_settings=settings,
        )


class CSVWriter(Writer):
    """Writer a dataframe to a csv file.

    Parameters:
        path (str): local or S3 path to the dataset.
    """

    KIND: T.Literal["CSVWriter"] = "CSVWriter"

    path: str
    header: bool = True
    index: bool = False

    @TX.override
    def write(self, data: pd.DataFrame) -> None:
        pd.DataFrame.to_csv(data, self.path, header=self.header, index=self.index)


class DeltalakeWriter(Writer):
    """Write a dataframe to the delta lake.

    Parameters:
        path (str): S3 path to the delta lake file.
    """

    KIND: T.Literal["DeltalakeWriter"] = "DeltalakeWriter"

    path: str

    @TX.override
    def write(self, data: pd.DataFrame) -> None:
        # s3_allow_unsafe_rename is required whever the file exists or not
        wr.s3.to_deltalake(df=data, path=self.path, s3_allow_unsafe_rename=True)


WriterKind = ParquetWriter | GlueWriter | DeltalakeWriter | CSVWriter
