import io

import pandas as pd

from business_entity_resolution.data.schema import READ_KWARGS
from business_entity_resolution.preprocessing.normalization import (
    basic_normalize,
    basic_normalize_series,
    to_ascii,
)

TSV = (
    "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    'S2-1\tJoe\'s "Best" Diner\t12 Main St, NA\tUS\n'
    "S2-2\tNA\t\tIndia\n"
    'S2-3\t"Quoted\tstart only\tUS\n'
)


def test_quotes_and_na_are_preserved():
    df = pd.read_csv(io.StringIO(TSV), **{k: v for k, v in READ_KWARGS.items() if k != "encoding"})
    assert len(df) == 3
    assert df.loc[0, "business_name"] == 'Joe\'s "Best" Diner'
    assert df.loc[1, "business_name"] == "NA"          # not converted to NaN
    assert df.loc[1, "business_address"] == ""         # empty string, not NaN
    assert df.loc[2, "business_name"] == '"Quoted'     # unbalanced quote does not swallow rows


def test_basic_normalize():
    assert basic_normalize("Apex Healthcare Pvt. Ltd.") == "apex healthcare pvt ltd"
    assert basic_normalize("Château  Mérignac") == "chateau merignac"
    assert to_ascii("लिमिटेड") == "limited"


def test_series_matches_scalar():
    s = pd.Series(["A & B Co.", "Café Délice", "एपेक्स", ""])
    assert basic_normalize_series(s).tolist() == [basic_normalize(x) for x in s]
