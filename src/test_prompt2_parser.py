import os
import sys
import tempfile

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.export_codabench_submission import parse_query_file
from src.search.object_search import ObjectSearcher


def write_query_file(folder, filename, content):
    path = os.path.join(folder, filename)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(content)

    return path


def test_kis(tmp_dir):
    content = (
        "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, "
        "cùng thực hiện động tác hai tay chạm mũi chân. "
        "Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ."
    )

    path = write_query_file(
        tmp_dir,
        "query-p1-1-kis.txt",
        content,
    )

    parsed = parse_query_file(path)

    assert parsed["task_type"] == "kis"
    assert parsed["query"] == content
    assert "events" not in parsed
    assert "question" not in parsed

    print("[PASS] KIS parser")


def test_qa(tmp_dir):
    content = (
        "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh "
        "một con cá khác cùng loại bị một người cầm đuôi. "
        "Con số hiển thị cuối cùng trên cân là bao nhiêu?"
    )

    expected_query = (
        "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh "
        "một con cá khác cùng loại bị một người cầm đuôi."
    )

    expected_question = (
        "Con số hiển thị cuối cùng trên cân là bao nhiêu?"
    )

    path = write_query_file(
        tmp_dir,
        "query-p1-3-qa.txt",
        content,
    )

    parsed = parse_query_file(path)

    assert parsed["task_type"] == "qa"
    assert parsed["query"] == expected_query
    assert parsed["question"] == expected_question

    print("[PASS] Q&A parser")


def test_official_trake(tmp_dir):
    content = (
        "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, "
        "mũi đỏ, bên cạnh lá cờ trắng viền đỏ.\n"
        "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng "
        "đang xoay vòng.\n"
        "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên "
        "các thanh trụ (thời điểm đâu tiên các chân của lân đặt trên "
        "trụ sau khi xoay).\n"
        "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân."
    )

    expected_context = (
        "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, "
        "mũi đỏ, bên cạnh lá cờ trắng viền đỏ."
    )

    expected_events = [
        (
            "Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng "
            "đang xoay vòng."
        ),
        (
            "Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên "
            "các thanh trụ (thời điểm đâu tiên các chân của lân đặt trên "
            "trụ sau khi xoay)."
        ),
        (
            "Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân."
        ),
    ]

    path = write_query_file(
        tmp_dir,
        "query-p1-16-trake.txt",
        content,
    )

    parsed = parse_query_file(path)

    assert parsed["task_type"] == "trake"
    assert parsed["context"] == expected_context

    # Context chinh la video retrieval query.
    assert parsed["query"] == expected_context

    # Chi E1/E2/E3 la semantic events.
    assert parsed["events"] == expected_events
    assert len(parsed["events"]) == 3

    assert expected_context not in parsed["events"]

    print("[PASS] Official TRAKE parser")


def test_fallback_trake_detection(tmp_dir):
    content = (
        "Một đoạn video trình diễn múa lân.\n"
        "E1 Hai con rồng xuất hiện.\n"
        "E2 Con lân xoay trên trụ.\n"
        "E3 Dùi chạm kẻng."
    )

    path = write_query_file(
        tmp_dir,
        "query-without-task-suffix.txt",
        content,
    )

    parsed = parse_query_file(path)

    assert parsed["task_type"] == "trake"
    assert len(parsed["events"]) == 3

    print("[PASS] TRAKE fallback detection")


def test_object_logic():
    config = {
        "data": {
            "objects_dir": "__objects_not_required_for_unit_test__"
        }
    }

    obj = ObjectSearcher(config)

    # Tier 1 hien la 3.5, nguong logic phai la >= 3.0.
    assert obj.get_entity_information_weight("lion") >= 3.0

    english_entities = set(
        obj.extract_target_entities(
            "a man holding a cell phone beside a camera lens"
        )
    )

    assert "cell phone" in english_entities
    assert "camera lens" in english_entities

    print("[PASS] ObjectSearcher tier + English phrase matching")


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_kis(tmp_dir)
        test_qa(tmp_dir)
        test_official_trake(tmp_dir)
        test_fallback_trake_detection(tmp_dir)

    test_object_logic()

    print("=" * 60)
    print("PROMPT 2 PARSER / LOGIC TESTS: ALL PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()