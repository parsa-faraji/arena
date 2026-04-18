from __future__ import annotations

from pathlib import Path

from arena.store import Case, Run, Variant, create_engine, init_db, session


def test_roundtrip(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)

    with session(engine) as s:
        v = Variant(name="v0", prompt="hi", model="gpt-4o-mini")
        s.add(v)
        s.commit()
        s.refresh(v)
        c = Case(dataset="demo", inputs_json='{"x": 1}', expected_json='{"y": 2}')
        r = Run(variant_id=v.id, dataset="demo", status="running", total_cases=1)
        s.add(c)
        s.add(r)
        s.commit()
        s.refresh(c)
        s.refresh(r)

    with session(engine) as s:
        got = s.get(Run, r.id)
        assert got is not None
        assert got.variant_id == v.id
        case = s.get(Case, c.id)
        assert case is not None
        assert case.inputs == {"x": 1}
        assert case.expected == {"y": 2}
