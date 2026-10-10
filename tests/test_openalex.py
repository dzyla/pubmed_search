import os
import sys

import pytest

pytest.importorskip("sentence_transformers")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "update_database"))
import openalex_update as oa  # noqa: E402

BODY = ("We measured the effect of the drug on tumour growth in mice and found that it was reduced by half "
        "in the treated group compared with the controls. ") * 5


def inverted(text):
    inv = {}
    for i, w in enumerate(text.split()):
        inv.setdefault(w, []).append(i)
    return inv


def work(title="Tumour growth in treated mice", abstract=BODY, **extra):
    return {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/x", "title": title, "type": "article",
            "publication_date": "1999-05-01", "abstract_inverted_index": inverted(abstract),
            "primary_location": {"source": {"display_name": "J Test"}}, "authorships": [], **extra}


def test_copyright_line_is_trimmed_not_rejected():
    row, reason = oa.to_row(work(abstract=BODY + "Copyright © 1999 John Wiley & Sons, Ltd. All rights reserved."))
    assert reason is None and row["abstract"].endswith("controls.") and "©" not in row["abstract"]


def test_landing_page_junk_short_and_non_papers_are_rejected():
    assert oa.to_row(work(abstract="ADVERTISEMENT RETURN TO ISSUE PREV Article " + BODY))[1] == "junk"
    assert oa.to_row(work(abstract="Too short to be an abstract."))[1] == "short"
    assert oa.to_row(work(title="Erratum"))[1] == "title"
    assert oa.to_row(work(title="Hypokalemia"))[1] is None            # one-word titles are fine


def test_meeting_abstracts_are_labelled():
    row, _ = oa.to_row(work(biblio={"issue": "Supplement_1", "first_page": "MP02-07"}))
    assert row["pub_type"] == "Conference abstract"
    assert oa.to_row(work(biblio={"issue": "3", "first_page": "e1234"}))[0]["pub_type"] == "Journal Article"


def test_rebuild_abstract_strips_label_and_markup():
    assert oa.rebuild_abstract(inverted("Abstract: The <i>cat</i> sat.")) == "The cat sat."
