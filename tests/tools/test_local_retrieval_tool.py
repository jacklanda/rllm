import importlib.util
from pathlib import Path


def _load_local_retrieval_tool():
    path = Path(__file__).resolve().parents[2] / "examples" / "search" / "local_retrieval_tool.py"
    spec = importlib.util.spec_from_file_location("_test_local_retrieval_tool", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.LocalRetrievalTool


LocalRetrievalTool = _load_local_retrieval_tool()


def test_local_retrieval_formats_unique_raw_evidence():
    tool = LocalRetrievalTool(server_url="http://127.0.0.1:9", max_results=3)
    results = [
        {
            "title": "Inter-stimulus interval",
            "document": "Inter-stimulus interval is the temporal interval between two stimuli in an experiment. "
            "It is often abbreviated as ISI and appears in psychology and neuroscience protocols.",
        },
        {
            "title": "Inter-stimulus interval",
            "document": "Inter-stimulus interval is the temporal interval between two stimuli in an experiment. "
            "It is often abbreviated as ISI and appears in psychology and neuroscience protocols.",
        },
        {
            "title": "Rilpivirine",
            "document": "Rilpivirine is an antiretroviral drug, commonly abbreviated RPV, used in combination therapy. "
            "The medication is discussed in HIV treatment guidelines and pharmacology references, where the abbreviation "
            "appears alongside dosage, resistance, and clinical trial evidence for identifying the compound.",
        },
    ]

    documents, metadata = tool._format_search_results(results, "isi rpv")

    assert documents[0].startswith("[Result 1] Title: Inter-stimulus interval")
    assert documents[1].startswith("[Result 2] Title: Rilpivirine")
    assert metadata["num_duplicates"] == 1
    assert metadata["num_unique"] == 2


def test_local_retrieval_low_information_results_trigger_rewrite_instruction():
    tool = LocalRetrievalTool(server_url="http://127.0.0.1:9", max_results=3)
    documents, metadata = tool._format_search_results(
        [
            {"title": "A", "document": "Dingxin Zhao"},
            {"title": "B", "document": "Xiangming Chen"},
        ],
        "Townizing China",
    )

    assert len(documents) == 1
    assert "Do not submit" in documents[0]
    assert "Rewrite the query" in documents[0]
    assert metadata["num_unique"] == 0
    assert metadata["num_short_filtered"] == 2
