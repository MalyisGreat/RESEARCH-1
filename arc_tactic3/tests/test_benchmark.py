from arc_tactic3.benchmark import run_benchmark


def test_benchmark_ranking_matches_design_intent() -> None:
    report = run_benchmark().to_dict()
    assert report["tactic"]["score"] > report["tactic_no_transfer"]["score"]
    assert report["tactic"]["score"] > report["tactic_no_planner"]["score"]
    assert report["tactic_no_transfer"]["score"] > report["frontier_graph"]["score"]
    assert report["frontier_graph"]["score"] >= report["random"]["score"]
    assert report["tactic"]["transfer_gain"] > report["tactic_no_transfer"]["transfer_gain"]
    assert report["tactic"]["robustness"]["button_order_reversed"] >= 0.95 * report["tactic"]["score"]
    assert report["tactic"]["robustness"]["button_labels_renamed"] >= 0.95 * report["tactic"]["score"]
