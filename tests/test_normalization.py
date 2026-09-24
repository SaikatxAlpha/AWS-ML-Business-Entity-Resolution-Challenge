from collections import Counter

import pandas as pd

from business_entity_resolution.preprocessing.address_normalization import prenormalize_addresses
from business_entity_resolution.preprocessing.name_normalization import core_name, prenormalize_names, suffix_tokens
from business_entity_resolution.preprocessing.token_maps import accept_substitutions, build_map
from business_entity_resolution.preprocessing.tokenization import abbreviation_compatible, fix_ocr_token, numbers


def test_ocr_repair_only_touches_alpha_tokens():
    assert fix_ocr_token("nati0nal") == "national"
    assert fix_ocr_token("8rothers") == "brothers"
    assert fix_ocr_token("5ervices") == "services"
    assert fix_ocr_token("b13") == "b13"      # only one letter
    assert fix_ocr_token("24hr") == "24hr"    # 2 is not OCR-confusable
    assert fix_ocr_token("1234") == "1234"
    assert fix_ocr_token("6roup") == "group"
    for ordinal in ("1st", "2nd", "3rd", "4th", "58th"):
        assert fix_ocr_token(ordinal) == ordinal


def test_name_prenormalization():
    s = pd.Series(["Joe's Pizza L.L.C.", "A & B Traders Pvt. Ltd.", "NATI0NAL CLÍNIC"])
    assert prenormalize_names(s).tolist() == ["joes pizza llc", "a and b traders pvt ltd", "national clinic"]


def test_address_prenormalization():
    s = pd.Series(["#003623 Beverly Ave, null, Salem, OR"])
    assert prenormalize_addresses(s).tolist() == ["3623 beverly ave salem or"]
    assert numbers("003623 Beverly, Unit 04") == ["3623", "4"]


def test_abbreviation_rule():
    assert abbreviation_compatible("rd", "road")
    assert abbreviation_compatible("mharastr", "maharashtra")
    assert abbreviation_compatible("tx", "texas")
    assert not abbreviation_compatible("shri", "limited")
    assert not abbreviation_compatible("r", "rue")          # single letters excluded


def test_substitution_acceptance_and_canonical_form():
    subs = Counter({("rd", "road"): 100, ("center", "private"): 90, ("center", "limited"): 80,
                    ("shri", "limited"): 60, ("ltd", "limited"): 50, ("limited", "ltd"): 30})
    acc = accept_substitutions(subs, min_count=25, min_consistency=0.5, min_ratio=70)
    got = {(m, s) for m, s, _ in acc}
    assert ("rd", "road") in got and ("ltd", "limited") in got
    assert ("center", "private") not in got      # inconsistent generic insertion
    assert ("shri", "limited") not in got         # not string-plausible
    mapping = build_map(acc, Counter({"limited": 10, "ltd": 3, "road": 5}))
    assert mapping["ltd"] == "limited" and mapping["rd"] == "road" and "limited" not in mapping


def test_weak_tokens():
    names = pd.Series(["alpha sarl", "beta sarl", "gamma sas", "delta club sarl", "sarl"] * 100)
    assert {"sarl", "sas"} <= suffix_tokens(names)
    assert core_name("the alpha club sarl", {"sarl"}) == "alpha club"
    assert core_name("sarl", {"sarl"}) == "sarl"   # never empty
